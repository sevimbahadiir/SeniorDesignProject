from locust import HttpUser, task, between

class RobotShopUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def view_products(self):
        self.client.get("/")
        self.client.get("/api/catalogue/categories")

    @task(3)
    def add_to_cart(self):
        # Ürünü sepete ekleme simülasyonu
        self.client.get("/api/cart/add/123")

    @task(1)
    def checkout(self):
        # Makaledeki kritik gecikme metriğine benzer ödeme işlemi
        self.client.get("/api/payment/pay/user123")