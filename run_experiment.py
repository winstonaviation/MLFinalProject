import pandas as pd
from preprocessing import load_and_split, build_tfidf, transform
from models import get_models
from evaluation import evaluate_model, print_results

#necessary processing otherwise the dataset is too large to load into memory. Only need to do this once, so it's commented out after creating the smaller sampled dataset.
'''
sampled_chunks = []
for chunk in pd.read_csv("reddit-dataset-3.csv", chunksize=10000, low_memory=False):
    sampled_chunks.append(chunk.sample(frac=0.02, random_state=42))
depression = pd.concat(sampled_chunks, ignore_index=True)
print(f"Depression rows: {len(depression)}")
depression.to_csv("depression-sampled.csv", index=False)
df = pd.read_csv("depression-sampled.csv", low_memory=False)
'''
# drop rows with null body, title, or label
df = df.dropna(subset=["body", "label"])

# fill null titles with empty string (less critical than body)
df["title"] = df["title"].fillna("")

df["label"] = df["label"].astype(int)

df.to_csv("depression-sampled.csv", index=False)

DATASETS = [
    {
        "name": "Depression & Reddit",
        "path": "reddit-dataset-3.csv",
        "text_col": "body",  
        "extra_text_cols": ["title"],  
        "label_col": "label",  
        "binary": True
    },
]

all_results = []

for dataset in DATASETS:
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset['name']}")
    print(f"{'='*50}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(
        dataset["path"], dataset["text_col"], dataset["label_col"],
        extra_text_cols=dataset.get("extra_text_cols")
    )
    #fit TF-IDF on training data only
    vectorizer, X_train_tfidf = build_tfidf(X_train)
    #print(len(vectorizer.vocabulary_))
    X_val_tfidf = transform(vectorizer, X_val)
    X_test_tfidf = transform(vectorizer, X_test)

    models = get_models()

    for model_name, model in models.items():
        #train
        model.fit(X_train_tfidf, y_train)

        #validate
        val_preds = model.predict(X_val_tfidf)

        #eval on test
        results = evaluate_model(
            model, X_test_tfidf, y_test,
            model_name=model_name,
            binary=dataset["binary"]
        )
        results["dataset"] = dataset["name"]
        all_results.append(results)
        print_results(results)

#save df
summary = pd.DataFrame([{
    "Dataset": r["dataset"],
    "Model": r["model"],
    "F1": r["f1"],
    "Accuracy": r["accuracy"],
    "AUROC": r.get("auroc", "N/A"),
    "Inference Time (s)": r["inference_time_sec"]
} for r in all_results])

summary.to_csv("results/classical2_results.csv", index=False)