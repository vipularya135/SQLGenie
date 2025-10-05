import sqlite3
import os

# Create sample company database
db_path = r'c:\Users\krish\OneDrive\Desktop\sharp\sample_company.db'

# Remove existing database if it exists
if os.path.exists(db_path):
    os.remove(db_path)

# Create connection and database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    manager_id INTEGER,
    budget REAL NOT NULL,
    FOREIGN KEY (manager_id) REFERENCES employees(id)
)
''')

cursor.execute('''
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT,
    budget REAL NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
)
''')

# Insert sample data
employees_data = [
    ('John Smith', 'Engineering', 75000, '2022-01-15'),
    ('Sarah Johnson', 'Marketing', 65000, '2022-03-01'),
    ('Mike Davis', 'Engineering', 80000, '2021-11-20'),
    ('Emily Brown', 'HR', 60000, '2023-02-10'),
    ('David Wilson', 'Engineering', 90000, '2020-08-15')
]

cursor.executemany('''
INSERT INTO employees (name, department, salary, hire_date) VALUES (?, ?, ?, ?)
''', employees_data)

departments_data = [
    ('Engineering', 5, 500000),
    ('Marketing', 2, 200000),
    ('HR', 4, 150000)
]

cursor.executemany('''
INSERT INTO departments (name, manager_id, budget) VALUES (?, ?, ?)
''', departments_data)

projects_data = [
    ('Mobile App Development', 1, '2023-01-01', '2023-12-31', 300000),
    ('Website Redesign', 2, '2023-06-01', '2023-09-30', 50000),
    ('Employee Training Program', 3, '2023-03-01', None, 75000),
    ('AI Integration', 1, '2023-09-01', None, 200000)
]

cursor.executemany('''
INSERT INTO projects (name, department_id, start_date, end_date, budget) VALUES (?, ?, ?, ?, ?)
''', projects_data)

# Commit changes and close connection
conn.commit()
conn.close()

print(f"Sample company database created successfully at: {db_path}")
print("Tables created: employees, departments, projects")
print("Sample data inserted for testing.")