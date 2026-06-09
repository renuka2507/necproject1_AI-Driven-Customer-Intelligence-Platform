def recommend_products(budget):
    if budget > 60000:
        return ["Smartphone","Laptop","Gaming Console"]

    elif budget > 25000:
        return ["Watch","Shoes","Headphones"]

    else:
        return ["T-Shirt","Grocery Essentials","Milk Pack"]
    