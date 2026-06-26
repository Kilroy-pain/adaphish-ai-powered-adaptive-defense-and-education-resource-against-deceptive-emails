import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import random

# Dummy phishing email dataset
class PhishingDataset(Dataset):
    def __init__(self, emails, labels):
        self.emails = emails
        self.labels = labels

    def __len__(self):
        return len(self.emails)

    def __getitem__(self, idx):
        return self.emails[idx], self.labels[idx]

# Simple LSTM model for phishing email classification
class PhishingClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PhishingClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.softmax(out)

# Function to anonymize emails
def anonymize_email(email):
    # Replace email addresses and URLs with placeholders
    email = email.replace("@", " [AT] ").replace(".", " [DOT] ")
    email = " ".join(["[URL]" if "http" in word else word for word in email.split()])
    return email

# Function to vectorize emails using TF-IDF
def vectorize_emails(emails):
    vectorizer = TfidfVectorizer(max_features=100)
    vectors = vectorizer.fit_transform(emails).toarray()
    return vectors, vectorizer

# Function to detect phishing using cosine similarity
def detect_phishing(email_vector, known_phishing_vectors, threshold=0.8):
    similarities = cosine_similarity([email_vector], known_phishing_vectors)
    max_similarity = np.max(similarities)
    return max_similarity > threshold, max_similarity

if __name__ == '__main__':
    # Dummy data
    emails = [
        "Dear user, your account has been compromised. Click http://phishing.com to secure it.",
        "Congratulations! You've won a $1000 gift card. Claim it at http://scam.com.",
        "Your invoice is ready. Please visit http://legitbusiness.com to view it.",
        "Reminder: Your subscription is expiring. Renew at http://phishing.com."
    ]
    labels = ["phishing", "phishing", "legitimate", "phishing"]

    # Anonymize emails
    anonymized_emails = [anonymize_email(email) for email in emails]

    # Vectorize emails
    email_vectors, vectorizer = vectorize_emails(anonymized_emails)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Create dataset and dataloader
    dataset = PhishingDataset(email_vectors, encoded_labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model, loss, and optimizer
    input_dim = email_vectors.shape[1]
    hidden_dim = 16
    output_dim = len(set(labels))
    model = PhishingClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        for batch_emails, batch_labels in dataloader:
            batch_emails = torch.tensor(batch_emails, dtype=torch.float32)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(batch_emails)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Simulate real-time phishing detection
    test_email = "Urgent: Your account is locked. Visit http://phishing.com to unlock it."
    test_email_anonymized = anonymize_email(test_email)
    test_email_vector = vectorizer.transform([test_email_anonymized]).toarray()[0]

    known_phishing_vectors = email_vectors[np.array(labels) == "phishing"]
    is_phishing, similarity = detect_phishing(test_email_vector, known_phishing_vectors)

    print(f"Test Email: {test_email}")
    print(f"Anonymized Email: {test_email_anonymized}")
    print(f"Is Phishing: {is_phishing}, Similarity: {similarity}")