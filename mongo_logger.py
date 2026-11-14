
import os
from pymongo import MongoClient
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


class MongoLogger:

    def __init__(self):

        # Connect to MongoDB
        mongo_uri = os.getenv(
            "MONGO_URI",
            "mongodb://localhost:27017/"
        )

        self.client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000
        )

        # Verify MongoDB connection
        self.client.admin.command("ping")

        # Select database and collection
        self.db = self.client["llm_monitoring"]
        self.collection = self.db["call_logs"]

        print("Connected to MongoDB successfully!")

    def log_call(self, log_data):

        # Add UTC timestamp
        log_data["timestamp"] = datetime.now(timezone.utc)

        # Insert log into MongoDB
        self.collection.insert_one(log_data)

        print("Call details saved in MongoDB!")

    def close(self):

        # Close MongoDB connection
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")
