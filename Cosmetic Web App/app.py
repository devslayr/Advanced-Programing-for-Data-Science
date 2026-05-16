import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

from flask import Flask, render_template, request, abort

app = Flask(__name__)

# Get current project folder path
BASE_DIR = Path(__file__).resolve().parent

# Path to your products.csv file
PRODUCTS_CSV = BASE_DIR / "data" / "products.csv"


# Temporary product images because products.csv does not have image column
# This is okay for display purpose because assignment allows artificial data/images.
IMAGE_POOL = [
    "https://images.unsplash.com/photo-1620916566398-39f1143ab7be",
    "https://images.unsplash.com/photo-1596462502278-27bfdc403348",
    "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9",
    "https://images.unsplash.com/photo-1631214524049-0ebbbe6d81aa",
    "https://images.unsplash.com/photo-1612817288484-6f916006741a",
    "https://images.unsplash.com/photo-1608248597279-f99d160bfcbc",
]


def get_product_image(index):
    """
    Assign one display image to each product.
    This is needed because the CSV has no product image column.
    """
    return IMAGE_POOL[index % len(IMAGE_POOL)]


def safe_float(value, default=0.0):
    """
    Convert price/rating safely.
    Prevents the website from crashing if the CSV has missing values.
    """
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except ValueError:
        return default


def safe_int(value, default=0):
    """
    Convert product IDs safely.
    This prevents one bad CSV row from breaking all product browsing.
    """
    try:
        if value == "" or value is None:
            return default
        return int(value)
    except ValueError:
        return default


def normalize_text(value):
    """
    Prepare text for searching.
    Lowercase text and remove punctuation so searches like "L'Oreal",
    "loreal", and "l oreal" are treated more consistently.
    """
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(first_text, second_text):
    """
    Return a similarity score from 0.0 to 1.0.
    This is used for typo-tolerant matching such as "Maybeline"
    matching "Maybelline".
    """
    return SequenceMatcher(None, first_text, second_text).ratio()


def meaningful_words(text):
    """
    Keep useful query words and ignore very short words that add noise.
    """
    return [word for word in normalize_text(text).split() if len(word) >= 3]


def word_matches_text(word, text, threshold=0.84):
    """
    Check if one query word appears in text, allowing small typos.
    """
    normalized_text = normalize_text(text)

    if word in normalized_text:
        return True

    return any(
        text_similarity(word, text_word) >= threshold
        for text_word in normalized_text.split()
    )


def word_matches_brand(word, brand_name):
    """
    Check if a query word belongs to the brand name.
    """
    brand_words = meaningful_words(brand_name)

    return any(
        word in brand_word
        or brand_word in word
        or text_similarity(word, brand_word) >= 0.84
        for brand_word in brand_words
    )


def extra_product_words(query, brand_name):
    """
    Return query words that are not part of the brand name.
    For "olay ultimate", "ultimate" is the extra product word.
    For "maybeline new york", there are no extra product words because the
    whole query is treated as a fuzzy brand search.
    """
    return [
        word for word in meaningful_words(query)
        if not word_matches_brand(word, brand_name)
    ]


def product_matches_extra_words(query, product):
    """
    If users search for brand + keyword, require the keyword part to match
    product-specific text. This prevents "olay ultimate" from returning every
    Olay product.
    """
    extra_words = extra_product_words(query, product["brand_name"])

    if not extra_words:
        return True

    product_text = (
        product["product_name"] + " " +
        product["product_tags"] + " " +
        product["description"]
    )

    return all(word_matches_text(word, product_text) for word in extra_words)


def brand_matches_query(query, brand_name):
    """
    Check whether the query likely refers to a brand.
    This supports similar keyword forms, for example:
    - "Maybeline"
    - "maybeline New York"
    both matching "Maybelline New York".
    """
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
    """
    Decide whether the whole query is mainly a brand search.
    If every meaningful query word matches a brand word, searches like
    "Maybeline" and "maybeline New York" should return the same brand results.
    """
    query_words = [word for word in normalize_text(query).split() if len(word) >= 4]
    brand_words = [word for word in normalize_text(brand_name).split() if len(word) >= 4]

    if not query_words or not brand_words:
        return False

    for query_word in query_words:
        if not any(
            query_word in brand_word
            or brand_word in query_word
            or text_similarity(query_word, brand_word) >= 0.84
            for brand_word in brand_words
        ):
            return False

    return True


def find_brand_search(products, query):
    """
    Return the best matching brand name when the query appears to be a brand.
    Otherwise return None and let the general product search run.
    """
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
    """
    Score each product for the entered query.
    Higher scores appear first in search results.
    """
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
    """
    Run fuzzy product search and sort results by relevance.
    A result is included only if its relevance score is greater than zero.
    """
    matched_brand = find_brand_search(products, query)

    if matched_brand:
        return sorted(
            [
                product for product in products
                if product["brand_name"] == matched_brand
            ],
            key=lambda product: product["avg_product_rating"],
            reverse=True
        )

    scored_products = []

    for product in products:
        if not product_matches_extra_words(query, product):
            continue

        score = score_product(query, product)

        if score >= 20:
            scored_products.append((score, product["avg_product_rating"], product))

    scored_products.sort(key=lambda item: (item[0], item[1]), reverse=True)

    return [product for score, rating, product in scored_products]


def create_description(product):
    """
    Create a product detail description from the CSV information.
    The dataset does not have a full description column, so we use
    brand, product title and tags to generate a readable detail section.
    """
    brand = product["brand_name"]
    title = product["product_name"]
    tags = product["product_tags"]

    if tags and tags.lower() != "nan":
        return f"{title} by {brand}. This product is related to {tags}, making it suitable for shoppers browsing beauty and cosmetic items."

    return f"{title} by {brand}. A beauty and cosmetic product available in the GlowHaus product catalogue."


def load_products():
    """
    Load all products from data/products.csv and convert them into dictionaries
    that can be used by the HTML templates.
    """
    products = []

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


@app.route("/")
def home():
    products = load_products()

    return render_template(
        "index.html",
        products=products,
        page_title="Trending Beauty Products",
        query="",
        result_message=f"Showing {len(products)} cosmetics and beauty products."
    )


@app.route("/search")
def search():
    """
    Fuzzy search route.
    It supports partial matching and typo-tolerant brand matching, so similar
    keyword forms such as "Maybeline" and "maybeline New York" return relevant
    Maybelline New York products.
    """
    query = request.args.get("query", "").strip()
    products = load_products()

    if query:
        products = search_products(products, query)

    result_count = len(products)

    return render_template(
        "index.html",
        products=products,
        query=query,
        page_title="Search Results" if query else "Trending Beauty Products",
        result_message=f'{result_count} cosmetics and beauty products matched "{query}".'
        if query else f"Showing {result_count} cosmetics and beauty products."
    )


@app.route("/product/<int:id>")
def product_detail(id):
    products = load_products()

    product = None

    for item in products:
        if item["id"] == id:
            product = item
            break

    if product is None:
        abort(404)

    reviews = []

    # Simple temporary recommendation:
    # show products from the same brand.
    recommendations = [
        item for item in products
        if item["brand_name"] == product["brand_name"] and item["id"] != product["id"]
    ][:4]

    return render_template(
        "product.html",
        product=product,
        reviews=reviews,
        recommendations=recommendations
    )


@app.route("/review/<int:id>", methods=["GET", "POST"])
def create_review(id):
    prediction_label = "Buyer"

    return render_template(
        "review.html",
        prediction_label=prediction_label
    )


if __name__ == "__main__":
    app.run(debug=True)
