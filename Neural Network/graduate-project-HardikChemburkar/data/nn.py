import argparse
import datasets
import pandas as pd
import transformers
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

# Function to train a multi-label classification model using transformer embeddings
def run_training(
    model_save_path="model.keras",               # Path to save the trained model
    transformer_save_dir="transformer_model",    # Directory to save tokenizer and transformer
    train_csv="train.csv",                       # Training dataset path
    validation_csv="dev.csv",                    # Validation dataset path
    test_csv="test.csv",                         # Testing dataset path
    batch_size=32,
    learning_rate=2e-5,
    epochs=10,
    dropout=0.4,
    dense_layers=[1024, 512, 256, 128],
    smoothing=0.01,
    sequence_length=128,
    thresholds=None
):
    # Loading pre-trained tokenizer and transformer model
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
    base_model = transformers.TFAutoModel.from_pretrained("distilroberta-base")

    # Loading training and validation data using HuggingFace datasets
    dataset = datasets.load_dataset("csv", data_files={"train": train_csv, "validation": validation_csv})
    label_names = dataset["train"].column_names[1:]
    print("Detected Labels:", label_names)

    # Extracting label values into a single "labels" column for each row
    dataset = dataset.map(lambda ex: {"labels": [float(ex[label]) for label in label_names]})
    
    # Tokenize the text field with truncation and padding
    dataset = dataset.map(lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=sequence_length), batched=True)

    # Prepare each record for TensorFlow
    def prepare_sample(example):
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"]
        }

    # Convert HuggingFace datasets to TensorFlow-compatible format
    train_data = dataset["train"].map(prepare_sample).with_format("tensorflow", columns=["input_ids", "attention_mask", "labels"])
    val_data = dataset["validation"].map(prepare_sample).with_format("tensorflow", columns=["input_ids", "attention_mask", "labels"])

    # Create TensorFlow datasets
    train_tf = train_data.to_tf_dataset(columns=["input_ids", "attention_mask"], label_cols="labels", shuffle=True, batch_size=batch_size)
    val_tf = val_data.to_tf_dataset(columns=["input_ids", "attention_mask"], label_cols="labels", batch_size=batch_size)

    # === Define the Model Architecture === #
    input_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="attention_mask")

    # Get contextual embeddings from the transformer
    sequence_output = base_model(input_ids, attention_mask=attention_mask)[0]

    # Combine global average and max pooling for a robust sentence representation
    pooled_avg = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    pooled_max = tf.keras.layers.GlobalMaxPooling1D()(sequence_output)
    merged = tf.keras.layers.Concatenate()([pooled_avg, pooled_max])

    # Add multiple fully-connected layers with dropout for classification
    x = merged
    for units in dense_layers:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    # Final output layer with sigmoid activation for multi-label classification
    final_output = tf.keras.layers.Dense(len(label_names), activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=final_output)

    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=smoothing),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(name="micro_F1", multi_label=True)],
    )

    # Callbacks for better training control
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor="val_micro_F1", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_micro_F1", patience=3, mode="max", restore_best_weights=True),
    ]

    # Train the model
    history = model.fit(train_tf, validation_data=val_tf, epochs=epochs, callbacks=callbacks)

    # Log training performance
    print("\nModel training finished.")
    for idx, loss in enumerate(history.history["loss"]):
        print(f"Epoch {idx+1}: Loss = {loss:.4f}, Val Micro F1 = {history.history['val_micro_F1'][idx]:.4f}")

    # === Evaluation === #
    y_true = np.vstack([batch[1].numpy() for batch in val_tf])
    y_probs = model.predict(val_tf)

    # Use predefined or custom thresholds
    if thresholds is None:
        thresholds = np.array([0.45, 0.45, 0.45, 0.45, 0.25, 0.25, 0.45])  # adjust as per label order

    y_pred = (y_probs > thresholds).astype(int)

    print("\nF1 Score (per label):")
    for idx, label in enumerate(label_names):
        print(f"{label}: {f1_score(y_true[:, idx], y_pred[:, idx]):.4f}")

    # Save the fine-tuned transformer and tokenizer
    base_model.save_pretrained(transformer_save_dir)
    tokenizer.save_pretrained(transformer_save_dir)
    print("\nTransformer and tokenizer saved.")

# === Inference Function for Prediction === #
def run_prediction(
    model_weights="model.keras",
    transformer_dir="transformer_model",
    test_file="test.csv",
    sequence_length=128,
    dense_layers=[1024, 512, 256, 128],
    dropout=0.4,
    thresholds=None
):
    # Reload tokenizer and transformer from saved directory
    tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_dir)
    transformer_model = transformers.TFAutoModel.from_pretrained(transformer_dir)

    # Load the test CSV file
    df = pd.read_csv(test_file)
    dataset = datasets.Dataset.from_pandas(df)
    label_names = df.columns[1:]
    y_actual = df.iloc[:, 1:].values

    # Tokenize the text field
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=sequence_length), batched=True)

    # Format for TF
    def format_sample(example):
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"]
        }

    tf_data = dataset.map(format_sample).with_format("tensorflow", columns=["input_ids", "attention_mask"]).to_tf_dataset(
        columns=["input_ids", "attention_mask"], batch_size=32
    )

    # Define same model architecture as used during training
    input_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32, name="attention_mask")
    embed_output = transformer_model(input_ids, attention_mask=attention_mask)[0]

    pooled_avg = tf.keras.layers.GlobalAveragePooling1D()(embed_output)
    pooled_max = tf.keras.layers.GlobalMaxPooling1D()(embed_output)
    merged = tf.keras.layers.Concatenate()([pooled_avg, pooled_max])

    x = merged
    for units in dense_layers:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    output_layer = tf.keras.layers.Dense(len(label_names), activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output_layer)
    model.load_weights(model_weights)  # Load saved weights

    # Make predictions
    y_scores = model.predict(tf_data)

    if thresholds is None:
        thresholds = np.array([0.45, 0.45, 0.45, 0.45, 0.25, 0.25, 0.45])

    y_preds = (y_scores > thresholds).astype(int)

    # Align shapes if necessary
    if y_preds.shape != y_actual.shape:
        min_rows = min(y_preds.shape[0], y_actual.shape[0])
        min_cols = min(y_preds.shape[1], y_actual.shape[1])
        y_actual = y_actual[:min_rows, :min_cols]
        y_preds = y_preds[:min_rows, :min_cols]

    # Print evaluation metrics
    print(f"\nMicro F1: {f1_score(y_actual, y_preds, average='micro'):.4f}")
    print(f"Macro F1: {f1_score(y_actual, y_preds, average='macro'):.4f}")

    print("\nLabel-wise F1:")
    for i, label in enumerate(label_names[:y_actual.shape[1]]):
        print(f"{label}: {f1_score(y_actual[:, i], y_preds[:, i]):.4f}")

    # Display example predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(df))):
        print(f"{df['text'][i]}")
        print(f"Predicted: {y_preds[i]}, Actual: {y_actual[i]}\n")

    # Save predictions into CSV inside a zip
    for i, label in enumerate(label_names[:y_preds.shape[1]]):
        df[label] = y_preds[:, i]

    df.to_csv("submission.zip", index=False, compression=dict(method="zip", archive_name="submission.csv"))
    print("\nSaved predictions to submission.zip.")

# === CLI Argument Parsing === #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "predict"])  # Only allow these two commands
    args = parser.parse_args()

    if args.command == "train":
        run_training()
    elif args.command == "predict":
        run_prediction()