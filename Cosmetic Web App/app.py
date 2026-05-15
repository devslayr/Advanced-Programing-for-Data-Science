from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():

    products = [
        {
            "id": 1,
            "product_name": "Hydrating Glow Serum",
            "brand_name": "GlowHaus",
            "price": 39.99,
            "avg_product_rating": 4.8,
            "image": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be"
        }
    ]

    return render_template(
        'index.html',
        products=products
    )


@app.route('/product/<int:id>')
def product_detail(id):

    product = {
        "id": 1,
        "product_name": "Hydrating Glow Serum",
        "brand_name": "GlowHaus",
        "price": 39.99,
        "avg_product_rating": 4.8,
        "description": "A premium hydrating serum for glowing skin.",
        "image": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be"
    }

    reviews = []

    recommendations = []

    return render_template(
        'product.html',
        product=product,
        reviews=reviews,
        recommendations=recommendations
    )


@app.route('/review/<int:id>', methods=['GET', 'POST'])
def create_review(id):

    prediction_label = "Buyer"

    return render_template(
        'review.html',
        prediction_label=prediction_label
    )


if __name__ == '__main__':
    app.run(debug=True)