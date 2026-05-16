import csv
import re
import os
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from math import sqrt
from pathlib import Path

from flask import Flask, render_template, request, abort, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = "glowhaus_secure_key_secret_cho_session"  # Securely encrypts sessions

# Global review log stream for the ML Admin Dashboard
review_logs = []

# Project folder pathing setup
BASE_DIR = Path(__file__).resolve().parent
PRODUCTS_CSV = BASE_DIR / "data" / "products.csv"
REVIEWS_CSV = BASE_DIR / "data" / "reviews.csv"
USERS_CSV = BASE_DIR / "data" / "users.csv"
MODELS_DIR = BASE_DIR / "models"

REVIEW_FIELDS = [
    "review_id",
    "product_id",
    "user_id",
    "review_title",
    "review_rating",
    "review_text",
    "is_a_buyer",
    "predicted_label",
    "predicted_probability",
    "created_at",
]

IMAGE_POOL = [
    "https://images.unsplash.com/photo-1620916566398-39f1143ab7be",
    "https://images.unsplash.com/photo-1596462502278-27bfdc403348",
    "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9",
    "https://images.unsplash.com/photo-1631214524049-0ebbbe6d81aa",
    "https://images.unsplash.com/photo-1612817288484-6f916006741a",
    "https://images.unsplash.com/photo-1608248597279-f99d160bfcbc",
]

def get_product_image(index):
    return IMAGE_POOL[index % len(IMAGE_POOL)]

def safe_float(value, default=0.0):
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except ValueError:
        return default

def safe_int(value, default=0):
    try:
        if value == "" or value is None:
            return default
        return int(value)
    except ValueError:
        return default

def buyer_status_to_binary(status):
    return 1 if str(status).strip().lower() in {"verified buyer", "true", "1", "buyer"} else 0

def buyer_binary_to_status(value):
    return "Verified Buyer" if safe_int(value, 0) == 1 else "Guest User"

def normalize_text(value):
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def text_similarity(first_text, second_text):
    return SequenceMatcher(None, first_text, second_text).ratio()

def meaningful_words(text):
    return [word for word in normalize_text(text).split() if len(word) >= 3]

def word_matches_text(word, text, threshold=0.84):
    normalized_text = normalize_text(text)
    if word in normalized_text:
        return True
    return any(text_similarity(word, text_word) >= threshold for text_word in normalized_text.split())

def word_matches_brand(word, brand_name):
    brand_words = meaningful_words(brand_name)
    return any(word in brand_word or brand_word in word or text_similarity(word, brand_word) >= 0.84 for brand_word in brand_words)

def extra_product_words(query, brand_name):
    return [word for word in meaningful_words(query) if not word_matches_brand(word, brand_name)]

def product_matches_extra_words(query, product):
    extra_words = extra_product_words(query, product["brand_name"])
    if not extra_words:
        return True
    product_text = product["product_name"] + " " + product["product_tags"] + " " + product["description"]
    return all(word_matches_text(word, product_text) for word in extra_words)

def brand_matches_query(query, brand_name):
    query_text = normalize_text(query)
    brand_text = normalize_text(brand_name)
    if not query_text or not brand_text:
        return False
    if query_text in brand_text or brand_text in query_text:
        return True
    if text_similarity(query_text, brand_text) >= 0.72:
        return True
    query_words = [word for word in query_text.split() if len(word) >= 4]
    brand_words = [word for word in brand_text.split() if len(word) >= 4]
    for query_word in query_words:
        for brand_word in brand_words:
            if text_similarity(query_word, brand_word) >= 0.84:
                return True
    return False

def query_is_brand_like(query, brand_name):
    query_words = [word for word in normalize_text(query).split() if len(word) >= 4]
    brand_words = [word for word in normalize_text(brand_name).split() if len(word) >= 4]
    if not query_words or not brand_words:
        return False
    for query_word in query_words:
        if not any(query_word in brand_word or brand_word in query_word or text_similarity(query_word, brand_word) >= 0.84 for brand_word in brand_words):
            return False
    return True

def find_brand_search(products, query):
    brand_names = sorted({product["brand_name"] for product in products})
    best_brand = None
    best_score = 0
    for brand_name in brand_names:
        if query_is_brand_like(query, brand_name):
            score = text_similarity(normalize_text(query), normalize_text(brand_name))
            if score > best_score:
                best_brand = brand_name
                best_score = score
    return best_brand

def score_product(query, product):
    query_text = normalize_text(query)
    brand_text = normalize_text(product["brand_name"])
    name_text = normalize_text(product["product_name"])
    tags_text = normalize_text(product["product_tags"])
    description_text = normalize_text(product["description"])
    searchable_text = f"{brand_text} {name_text} {tags_text} {description_text}"

    if not query_text:
        return 0
    score = 0
    if brand_matches_query(query_text, brand_text):
        score += 100
    if query_text in name_text:
        score += 60
    if query_text in searchable_text:
        score += 40
    for query_word in query_text.split():
        if len(query_word) < 3:
            continue
        if query_word in brand_text:
            score += 25
        elif any(text_similarity(query_word, brand_word) >= 0.84 for brand_word in brand_text.split()):
            score += 20
        if query_word in name_text:
            score += 15
        elif any(text_similarity(query_word, title_word) >= 0.84 for title_word in name_text.split()):
            score += 10
        if query_word in tags_text:
            score += 8
    return score

def search_products(products, query):
    matched_brand = find_brand_search(products, query)
    if matched_brand:
        return sorted([product for product in products if product["brand_name"] == matched_brand], key=lambda product: product["avg_product_rating"], reverse=True)
    scored_products = []
    for product in products:
        if not product_matches_extra_words(query, product):
            continue
        score = score_product(query, product)
        if score >= 20:
            scored_products.append((score, product["avg_product_rating"], product))
    scored_products.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [product for score, rating, product in scored_products]

def get_similarity_tokens(product):
    tags = product["product_tags"]
    if tags.lower() == "nan":
        tags = ""
    text = f"{product['brand_name']} {product['product_name']} {tags}"
    return [word for word in normalize_text(text).split() if len(word) >= 3]

def get_product_vector(product):
    return Counter(get_similarity_tokens(product))

def cosine_similarity(first_vector, second_vector):
    if not first_vector or not second_vector:
        return 0.0
    shared_tokens = set(first_vector) & set(second_vector)
    dot_product = sum(first_vector[token] * second_vector[token] for token in shared_tokens)
    first_length = sqrt(sum(value * value for value in first_vector.values()))
    second_length = sqrt(sum(value * value for value in second_vector.values()))
    if first_length == 0 or second_length == 0:
        return 0.0
    return dot_product / (first_length * second_length)

def get_similar_products(products, selected_product, limit=5):
    selected_vector = get_product_vector(selected_product)
    scored_products = []
    for product in products:
        if product["id"] == selected_product["id"]:
            continue
        product_vector = get_product_vector(product)
        similarity_score = cosine_similarity(selected_vector, product_vector)
        scored_products.append((similarity_score, product["avg_product_rating"], product))
    scored_products.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [product for score, rating, product in scored_products[:limit]]

def create_description(product):
    brand = product["brand_name"]
    title = product["product_name"]
    tags = product["product_tags"]
    if tags and tags.lower() != "nan":
        return f"{title} by {brand}. This product is related to {tags}, making it suitable for shoppers browsing beauty and cosmetic items."
    return f"{title} by {brand}. A beauty and cosmetic product available in the GlowHaus product catalogue."

def load_products():
    products = []
    if not os.path.exists(PRODUCTS_CSV):
        return products
    with open(PRODUCTS_CSV, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            product = {
                "id": safe_int(row.get("product_id"), index + 1),
                "brand_name": row.get("brand_name", "Unknown Brand"),
                "product_name": row.get("product_title", "Untitled Product"),
                "price": safe_float(row.get("price")),
                "avg_product_rating": safe_float(row.get("avg_product_rating")),
                "product_tags": str(row.get("product_tags", "")),
                "product_url": row.get("product_url", ""),
                "image": get_product_image(index),
            }
            product["description"] = create_description(product)
            products.append(product)
    return products

def find_product(product_id):
    return next((product for product in load_products() if product["id"] == product_id), None)

def ensure_reviews_file_schema():
    """
    Keep reviews.csv compatible with the ML review fields while preserving old rows.
    """
    if not os.path.exists(REVIEWS_CSV):
        with open(REVIEWS_CSV, mode="w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
        return

    with open(REVIEWS_CSV, mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        current_fields = reader.fieldnames or []
        rows = list(reader)

    if current_fields == REVIEW_FIELDS:
        return

    normalized_rows = []
    for row in rows:
        if not row:
            continue

        normalized_row = {field: row.get(field, "") for field in REVIEW_FIELDS}
        normalized_rows.append(normalized_row)

    with open(REVIEWS_CSV, mode="w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_rows)

def load_all_reviews():
    """Reads all review rows safely from reviews.csv."""
    reviews = []
    if not os.path.exists(REVIEWS_CSV):
        return reviews

    with open(REVIEWS_CSV, mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("review_id") == "review_id":
                continue

            reviews.append({
                "review_id": safe_int(row.get("review_id"), 0),
                "product_id": safe_int(row.get("product_id"), 0),
                "user_id": safe_int(row.get("user_id"), 0),  # Saved for matching author names
                "review_title": row.get("review_title", "No Title"),
                "review_rating": safe_float(row.get("review_rating"), 0.0),
                "review_text": row.get("review_text", ""),
                "is_a_buyer": row.get("is_a_buyer", "Guest User").strip(),
                "predicted_label": row.get("predicted_label", "").strip(),
                "predicted_probability": safe_float(row.get("predicted_probability"), 0.0),
                "created_at": row.get("created_at", "").strip(),
            })
    return reviews

def load_all_users():
    users = []
    if not os.path.exists(USERS_CSV):
        return users
    with open(USERS_CSV, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            users.append(row)
    return users

def save_new_user(username, password, email):
    existing_users = load_all_users()
    next_id = len(existing_users) + 1
    needs_newline = False
    if os.path.exists(USERS_CSV) and os.stat(USERS_CSV).st_size > 0:
        with open(USERS_CSV, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                needs_newline = True

    with open(USERS_CSV, mode="a", encoding="utf-8-sig", newline="") as file:
        if needs_newline:
            file.write("\n")
        fieldnames = ["user_id", "username", "password_hash", "email"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if os.stat(USERS_CSV).st_size == 0:
            writer.writeheader()
        writer.writerow({
            "user_id": next_id,
            "username": username,
            "password_hash": password,
            "email": email
        })

# =========================================================================
# MACHINE LEARNING CLASSIFICATION & TELEMETRY INITIALIZATION ENGINE
# =========================================================================
def load_text_file(path):
    if not path.exists():
        return set()

    with open(path, mode="r", encoding="utf-8") as file:
        return {
            line.strip().split(":")[0]
            for line in file
            if line.strip()
        }

def load_ml_artifacts():
    try:
        import joblib
        import numpy as np

        return {
            "available": True,
            "np": np,
            "model_desc": joblib.load(MODELS_DIR / "model_desc.joblib"),
            "model_title": joblib.load(MODELS_DIR / "model_title.joblib"),
            "model_num": joblib.load(MODELS_DIR / "model_num.joblib"),
            "title_tfidf": joblib.load(MODELS_DIR / "title_tfidf.joblib"),
            "glove_dict": joblib.load(MODELS_DIR / "glove_dict.joblib"),
            "stopwords": load_text_file(MODELS_DIR / "stopwords_en.txt"),
            "vocab": load_text_file(MODELS_DIR / "vocab.txt"),
            "error": "",
        }
    except Exception as error:
        return {"available": False, "error": str(error)}

ML_ARTIFACTS = load_ml_artifacts()

def class_is_buyer(value):
    return str(value).strip().lower() in {"1", "true", "buyer", "verified buyer", "yes"}

def get_buyer_probability(model, features):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = getattr(model, "classes_", None)

        if classes is not None:
            for index, class_value in enumerate(classes):
                if class_is_buyer(class_value):
                    return float(probabilities[index])

        return float(probabilities[-1])

    if hasattr(model, "decision_function"):
        np = ML_ARTIFACTS["np"]
        score = float(model.decision_function(features)[0])
        return float(1 / (1 + np.exp(-score)))

    prediction = model.predict(features)[0]
    return 1.0 if class_is_buyer(prediction) else 0.0

def get_glove_size(glove_dict):
    for vector in glove_dict.values():
        return len(vector)

    return 100

def preprocess_review_text(text, stopwords):
    return [
        word for word in normalize_text(text).split()
        if len(word) >= 2 and word not in stopwords
    ]

def review_text_to_glove_vector(text):
    np = ML_ARTIFACTS["np"]
    glove_dict = ML_ARTIFACTS["glove_dict"]
    stopwords = ML_ARTIFACTS["stopwords"]
    vocab = ML_ARTIFACTS["vocab"]
    embedding_size = get_glove_size(glove_dict)
    tokens = preprocess_review_text(text, stopwords)
    vectors = []

    for token in tokens:
        if vocab and token not in vocab:
            continue

        if token in glove_dict:
            vectors.append(np.asarray(glove_dict[token], dtype=float))

    if not vectors:
        return np.zeros((1, embedding_size))

    return np.mean(vectors, axis=0).reshape(1, -1)

def predict_review_with_models(review_title, review_text, product):
    if not ML_ARTIFACTS["available"]:
        raise RuntimeError(
            "ML models could not be loaded. Install requirements and check model files. "
            f"Original error: {ML_ARTIFACTS['error']}"
        )

    desc_features = review_text_to_glove_vector(review_text)
    title_features = ML_ARTIFACTS["title_tfidf"].transform([review_title])
    numeric_features = [[product["price"], product["avg_product_rating"]]]

    prob_desc = get_buyer_probability(ML_ARTIFACTS["model_desc"], desc_features)
    prob_title = get_buyer_probability(ML_ARTIFACTS["model_title"], title_features)
    prob_num = get_buyer_probability(ML_ARTIFACTS["model_num"], numeric_features)
    final_probability = (prob_desc + prob_title + prob_num) / 3.0
    predicted_label = "Verified Buyer" if final_probability >= 0.5 else "Guest User"
    confidence = final_probability if predicted_label == "Verified Buyer" else 1 - final_probability

    return {
        "prob_desc": prob_desc,
        "prob_title": prob_title,
        "prob_num": prob_num,
        "final_probability": final_probability,
        "predicted_label": predicted_label,
        "predicted_binary": 1 if predicted_label == "Verified Buyer" else 0,
        "confidence": confidence,
    }

def predict_buyer_status(review_title, review_text, rating, product=None):
    """
    Return the binary prediction used by the admin telemetry.
    Uses the real ML ensemble when product metadata is available.
    """
    if product is not None and ML_ARTIFACTS["available"]:
        return predict_review_with_models(review_title, review_text, product)["predicted_binary"]

    full_text = f"{review_title} {review_text}".lower()
    buyer_signals = ['buy', 'buying', 'bought', 'delivery', 'shipped', 'package', 'worth', 'oil control', 'matte', 'pigmented']
    promotional_signals = ['ad', 'sponsored', 'free sample', 'promotion', 'gifted', 'sticky']
    
    buyer_score = sum(1 for word in buyer_signals if word in full_text)
    promo_score = sum(1 for word in promotional_signals if word in full_text)
    
    if safe_float(rating) >= 4.0:
        buyer_score += 1
        
    if promo_score > buyer_score:
        return 0  # Non-Buyer Prediction
    return 1  # Buyer Prediction

def initialize_review_logs():
    """
    Populates global review_logs dynamically from historical reviews.csv file data,
    then adds dummy filler rows if totals are under 11 to immediately demo page pagination.
    """
    global review_logs
    ensure_reviews_file_schema()
    review_logs = []
    historical_reviews = load_all_reviews()
    product_map = {product["id"]: product for product in load_products()}
    
    for r in historical_reviews:
        pred = predict_buyer_status(
            r["review_title"],
            r["review_text"],
            r["review_rating"],
            product_map.get(r["product_id"]),
        )
        actual = buyer_status_to_binary(r["is_a_buyer"])
        review_logs.append({
            "transaction_id": r["review_id"] if r["review_id"] > 0 else 1000 + len(review_logs),
            "title": r["review_title"],
            "predicted": pred,
            "actual": actual,
            "overridden": (pred != actual)
        })
        
    # SEEDING BLOCK: Guarantees your table shows multiple pages right away if data is sparse
    if len(review_logs) < 11:
        mock_samples = [
            {"transaction_id": 101, "title": "Sponsored: decent product", "predicted": 0, "actual": 0, "overridden": False},
            {"transaction_id": 102, "title": "Received free promotional item", "predicted": 0, "actual": 0, "overridden": False},
            {"transaction_id": 103, "title": "Influenster sample review", "predicted": 0, "actual": 1, "overridden": True},
            {"transaction_id": 104, "title": "Great hydration booster", "predicted": 1, "actual": 1, "overridden": False},
            {"transaction_id": 105, "title": "Bait ad link click tracker", "predicted": 0, "actual": 0, "overridden": False},
            {"transaction_id": 106, "title": "Ad: Glowing serum experience", "predicted": 0, "actual": 0, "overridden": False},
            {"transaction_id": 107, "title": "Gifted by brand for evaluation", "predicted": 0, "actual": 0, "overridden": False},
            {"transaction_id": 108, "title": "Highly recommend to everyone", "predicted": 1, "actual": 1, "overridden": False},
        ]
        review_logs.extend(mock_samples)

# Run initial boot configuration parameters loading process
initialize_review_logs()

# =========================================================================
# FLASK WEB SERVER ROUTING INTERFACES
# =========================================================================
@app.route("/")
def home():
    all_products = load_products()
    brands = sorted(list(set(p["brand_name"] for p in all_products if p["brand_name"])))
    selected_brand = request.args.get("brand", "").strip()
    selected_sort = request.args.get("sort", "").strip()
    query = request.args.get("query", "").strip() 

    filtered_products = all_products
    if selected_brand:
        filtered_products = [p for p in filtered_products if p["brand_name"].lower() == selected_brand.lower()]

    if selected_sort == "price_low":
        filtered_products.sort(key=lambda x: x["price"] if x["price"] is not None else float('inf'))
    elif selected_sort == "price_high":
        filtered_products.sort(key=lambda x: x["price"] if x["price"] is not None else float('-inf'), reverse=True)
    elif selected_sort == "rating_high":
        filtered_products.sort(key=lambda x: x["avg_product_rating"] if x["avg_product_rating"] is not None else 0, reverse=True)

    result_message = f"Showing {len(filtered_products)} cosmetics and beauty products.\n"
    if selected_brand:
        result_message += f"\nFiltered by Brand: {selected_brand}.\n"

    sort_labels = {"price_low": "Price: Low to High", "price_high": "Price: High to Low", "rating_high": "Highest Rated"}
    if selected_sort in sort_labels:
        result_message += f"\nSorted by: {sort_labels[selected_sort]}."

    return render_template("index.html", products=filtered_products, brands=brands, selected_brand=selected_brand, selected_sort=selected_sort, page_title="Trending Beauty Products", query=query, result_message=result_message)

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    products = load_products()
    if query:
        products = search_products(products, query)
    result_count = len(products)
    return render_template("index.html", products=products, query=query, page_title="Search Results" if query else "Trending Beauty Products", result_message=f'{result_count} cosmetics products matched "{query}".' if query else f"Showing {result_count} cosmetics products.")

@app.route("/product/<int:id>")
def product_detail(id):
    products = load_products()
    product = next((item for item in products if item["id"] == id), None)
    if product is None:
        abort(404)

    all_reviews = load_all_reviews()
    all_users = load_all_users()
    user_map = {str(u["user_id"]): u["username"] for u in all_users}

    matched_reviews = []
    for r in all_reviews:
        if r["product_id"] == id:
            review_user_id = str(r.get("user_id", 0))
            r["reviewer_name"] = user_map.get(review_user_id, "Anonymous Guest")
            matched_reviews.append(r)

    recommendations = get_similar_products(products, product)
    return render_template("product.html", product=product, reviews=matched_reviews, recommendations=recommendations)

@app.route("/review/<int:id>", methods=["GET", "POST"])
def create_review(id):
    product = find_product(id)
    if product is None:
        abort(404)

    form_data = {
        "review_title": "",
        "review_text": "",
        "review_rating": "5",
    }
    prediction = None
    error_message = ""

    if request.method == "POST":
        step = request.form.get("step", "predict")
        form_data = {
            "review_title": request.form.get("review_title", "").strip(),
            "review_text": request.form.get("review_text", "").strip(),
            "review_rating": request.form.get("review_rating", "5"),
        }
        title = form_data["review_title"]
        text = form_data["review_text"]
        rating = form_data["review_rating"]

        if not title:
            error_message = "Please enter a review title."
        elif len(text) < 10:
            error_message = "Please enter a review description with at least 10 characters."
        elif safe_int(rating, 0) not in {1, 2, 3, 4, 5}:
            error_message = "Please select a valid rating from 1 to 5."

        if error_message:
            return render_template(
                "review.html",
                product=product,
                form_data=form_data,
                prediction=prediction,
                error_message=error_message,
            )

        if step == "predict":
            try:
                prediction = predict_review_with_models(title, text, product)
            except RuntimeError as error:
                error_message = str(error)

            return render_template(
                "review.html",
                product=product,
                form_data=form_data,
                prediction=prediction,
                error_message=error_message,
            )

        if step == "confirm":
            predicted_label = request.form.get("predicted_label", "")
            predicted_probability = safe_float(request.form.get("predicted_probability"), 0.0)
            buyer_status = request.form.get("is_a_buyer", predicted_label or "Guest User")
            current_user_id = session.get("user_id", 0)
            existing_reviews = load_all_reviews()
            next_review_id = max([review["review_id"] for review in existing_reviews] or [0]) + 1

            ensure_reviews_file_schema()
            with open(REVIEWS_CSV, mode="a", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
                writer.writerow({
                    "review_id": next_review_id,
                    "product_id": id,
                    "user_id": current_user_id,
                    "review_title": title,
                    "review_rating": float(rating),
                    "review_text": text,
                    "is_a_buyer": buyer_status,
                    "predicted_label": predicted_label,
                    "predicted_probability": f"{predicted_probability:.4f}",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

            ai_pred_binary = buyer_status_to_binary(predicted_label)
            user_actual_binary = buyer_status_to_binary(buyer_status)
            is_overridden = (ai_pred_binary != user_actual_binary)

            review_logs.append({
                "transaction_id": 1001 + len(review_logs),
                "title": title if title else "Untitled Review",
                "predicted": ai_pred_binary,
                "actual": user_actual_binary,
                "overridden": is_overridden
            })

            return redirect(url_for("product_detail", id=id))

    return render_template(
        "review.html",
        product=product,
        form_data=form_data,
        prediction=prediction,
        error_message=error_message,
    )

@app.route("/reviews/<int:review_id>")
def review_detail(review_id):
    review = next((item for item in load_all_reviews() if item["review_id"] == review_id), None)
    if review is None:
        abort(404)

    product = find_product(review["product_id"])
    users = load_all_users()
    user_map = {str(user["user_id"]): user["username"] for user in users}
    review["reviewer_name"] = user_map.get(str(review.get("user_id", 0)), "Anonymous Guest")

    return render_template("review_detail.html", review=review, product=product)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        users = load_all_users()
        if any(u["username"].lower() == username.lower() for u in users):
            flash("Username is already taken!", "error")
            return redirect(url_for("signup"))
        if any(u["email"].lower() == email.lower() for u in users):
            flash("This email is already registered!", "error")
            return redirect(url_for("signup"))
            
        save_new_user(username, password, email)
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = load_all_users()
        user = next((u for u in users if u["username"].lower() == username.lower()), None)
        if user and user["password_hash"] == password:
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password!", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("home"))

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin":
            session["is_authenticated"] = True
            session["admin_user"] = username
            return redirect(url_for("admin_dashboard"))
        else:
            error = "Invalid administrative credentials."
    return render_template("admin_login.html", error=error)

@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect("/")

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("is_authenticated"):
        return redirect(url_for("admin_login"))

    total_reviews = len(review_logs)
    if total_reviews == 0:
        stats = {"total": 0, "buyer_ratio": 0.0, "override_rate": 0.0, "accuracy": 100.0}
    else:
        buyers = sum(1 for x in review_logs if x['actual'] == 1)
        overrides = sum(1 for x in review_logs if x['overridden'])
        correct = sum(1 for x in review_logs if x['predicted'] == x['actual'])
        stats = {
            "total": total_reviews,
            "buyer_ratio": round((buyers / total_reviews) * 100, 1),
            "override_rate": round((overrides / total_reviews) * 100, 1),
            "accuracy": round((correct / total_reviews) * 100, 1)
        }

    try:
        page = int(request.args.get('page', 1))
        if page < 1: page = 1
    except ValueError:
        page = 1

    PER_PAGE = 10
    start_index = (page - 1) * PER_PAGE
    end_index = start_index + PER_PAGE
    
    paginated_logs = review_logs[::-1][start_index:end_index]
    total_pages = (total_reviews + PER_PAGE - 1) // PER_PAGE
    if total_pages == 0: total_pages = 1
        
    return render_template('admin_dashboard.html', stats=stats, logs=paginated_logs, current_page=page, total_pages=total_pages)

if __name__ == "__main__":
    app.run(debug=True)
