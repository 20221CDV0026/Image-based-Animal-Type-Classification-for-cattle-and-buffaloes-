import sqlite3

# Create DB + Table if not exists
def create_table():
    conn = sqlite3.connect("animals.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS animals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        animal_id TEXT,
        animal_type TEXT,
        breed TEXT,
        health_status TEXT,
        image_path TEXT
    )
    """)

    conn.commit()
    conn.close()


# Save prediction result
def save_animal(animal_id, animal_type, breed, health_status, image_path):
    conn = sqlite3.connect("animals.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO animals (animal_id, animal_type, breed, health_status, image_path)
        VALUES (?, ?, ?, ?, ?)
    """, (animal_id, animal_type, breed, health_status, image_path))

    conn.commit()
    conn.close()


# Fetch all saved animals (for dashboard later)
def get_all_animals():
    conn = sqlite3.connect("animals.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM animals")
    data = cursor.fetchall()

    conn.close()
    return data