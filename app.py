import hashlib
import os
import re
import types
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error as MySQLError
    mysql = types.SimpleNamespace(connector=mysql_connector)
except Exception:
    class _FallbackMySQLConnector:
        @staticmethod
        def connect(**kwargs):
            raise RuntimeError("MySQL is not available")

    mysql = types.SimpleNamespace(connector=_FallbackMySQLConnector)
    MySQLError = RuntimeError

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras_facenet import FaceNet
from mtcnn import MTCNN


class AttendanceApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance Dashboard")
        self.root.geometry("1280x760")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#f4f7fb")

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure("Sidebar.TFrame", background="#111827")
        self.style.configure("Content.TFrame", background="#f4f7fb")
        self.style.configure("Card.TFrame", background="#ffffff")
        self.style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), background="#f4f7fb", foreground="#0f172a")
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 11), background="#f4f7fb", foreground="#64748b")
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        self.db = self.connect_database()
        self.cursor = self.db.cursor()
        self.create_tables()
        os.makedirs("dataset", exist_ok=True)
        os.makedirs("trainer", exist_ok=True)

        self.face_model = None
        self.face_detector = None
        self.current_user = None
        self.auth_mode = "login"

        self.build_auth_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def connect_database(self):
        config = {
            "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", "root@123"),
        }
        database_name = os.getenv("MYSQL_DATABASE", "attendance_system")

        try:
            connection = mysql.connector.connect(**config)
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
            connection.database = database_name
            return connection
        except Exception as exc:
            raise RuntimeError("MySQL connection failed. Make sure MySQL is running and configured correctly.") from exc

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password_hash VARCHAR(64) NOT NULL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INT PRIMARY KEY,
                name VARCHAR(255) NOT NULL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INT PRIMARY KEY,
                name VARCHAR(255),
                date VARCHAR(20),
                time VARCHAR(20),
                status VARCHAR(20)
            )
        """)
        self.db.commit()

    def initialize_facenet_components(self):
        if self.face_model is None:
            self.face_model = FaceNet()
        if self.face_detector is None:
            self.face_detector = MTCNN()
        return self.face_model, self.face_detector

    def get_face_embedding(self, image, from_bgr=True):
        face_model, _ = self.initialize_facenet_components()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if from_bgr else image
        if rgb_image is None:
            return None

        try:
            result = face_model.extract(rgb_image, threshold=0.95)
        except Exception:
            return None

        if not result:
            return None
        return np.asarray(result[0]["embedding"], dtype=np.float32)

    def detect_faces(self, image):
        _, detector = self.initialize_facenet_components()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_image)
        faces = []

        for detection in detections:
            x, y, w, h = detection["box"]
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            faces.append((x, y, w, h))

        return faces

    def cosine_similarity(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if np.isclose(denominator, 0.0):
            return 0.0
        return float(np.dot(a, b) / denominator)

    def parse_student_id_from_filename(self, filename):
        match = re.search(r"User\.(\d+)\.", filename)
        if not match:
            return None
        return int(match.group(1))

    def hash_password(self, password):
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def build_auth_ui(self):
        self.auth_container = ttk.Frame(self.root, style="Content.TFrame")
        self.auth_container.pack(fill="both", expand=True)

        panel = tk.Frame(self.auth_container, bg="white", bd=1, relief="solid", padx=32, pady=32)
        panel.pack(expand=True, padx=40, pady=40)

        tk.Label(
            panel,
            text="FaceFlow Authentication",
            font=("Segoe UI", 22, "bold"),
            bg="white",
            fg="#111827"
        ).pack(anchor="w")
        tk.Label(
            panel,
            text="Sign in to access the attendance dashboard.",
            font=("Segoe UI", 11),
            bg="white",
            fg="#64748b"
        ).pack(anchor="w", pady=(6, 20))

        self.auth_mode_frame = tk.Frame(panel, bg="white")
        self.auth_mode_frame.pack(fill="x", pady=(0, 16))

        tk.Button(
            self.auth_mode_frame,
            text="Login",
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            command=lambda: self.switch_auth_mode("login")
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            self.auth_mode_frame,
            text="Register",
            bg="#e2e8f0",
            fg="#0f172a",
            font=("Segoe UI", 10, "bold"),
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            command=lambda: self.switch_auth_mode("register")
        ).pack(side="left")

        self.auth_form_frame = tk.Frame(panel, bg="white")
        self.auth_form_frame.pack(fill="x")

        self.auth_username_var = tk.StringVar()
        self.auth_password_var = tk.StringVar()
        self.auth_confirm_var = tk.StringVar()

        self.render_auth_form()

    def switch_auth_mode(self, mode):
        self.auth_mode = mode
        self.render_auth_form()

    def render_auth_form(self):
        for child in self.auth_form_frame.winfo_children():
            child.destroy()

        if self.auth_mode == "login":
            title = "Login"
            subtitle = "Enter your account details to continue"
            button_text = "Login"
            button_command = self.handle_login
            footer_text = "New here? Register an account"
            footer_command = lambda: self.switch_auth_mode("register")
            show_confirm = False
        else:
            title = "Create Account"
            subtitle = "Register a new account to access the dashboard"
            button_text = "Create Account"
            button_command = self.handle_register
            footer_text = "Already have an account? Login"
            footer_command = lambda: self.switch_auth_mode("login")
            show_confirm = True

        tk.Label(self.auth_form_frame, text=title, font=("Segoe UI", 16, "bold"), bg="white", fg="#111827").pack(anchor="w")
        tk.Label(self.auth_form_frame, text=subtitle, font=("Segoe UI", 10), bg="white", fg="#64748b").pack(anchor="w", pady=(4, 16))

        fields = tk.Frame(self.auth_form_frame, bg="white")
        fields.pack(fill="x")

        tk.Label(fields, text="Username", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=0, column=0, sticky="w", pady=8)
        tk.Entry(fields, textvariable=self.auth_username_var, font=("Segoe UI", 11), width=30).grid(row=0, column=1, padx=(12, 0), pady=8)

        tk.Label(fields, text="Password", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=1, column=0, sticky="w", pady=8)
        tk.Entry(fields, textvariable=self.auth_password_var, font=("Segoe UI", 11), width=30, show="*").grid(row=1, column=1, padx=(12, 0), pady=8)

        if show_confirm:
            tk.Label(fields, text="Confirm Password", font=("Segoe UI", 10, "bold"), bg="white", fg="#334155").grid(row=2, column=0, sticky="w", pady=8)
            tk.Entry(fields, textvariable=self.auth_confirm_var, font=("Segoe UI", 11), width=30, show="*").grid(row=2, column=1, padx=(12, 0), pady=8)

        tk.Button(
            self.auth_form_frame,
            text=button_text,
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=button_command
        ).pack(anchor="w", pady=(16, 10))

        tk.Button(
            self.auth_form_frame,
            text=footer_text,
            bg="white",
            fg="#2563eb",
            font=("Segoe UI", 10),
            bd=0,
            cursor="hand2",
            command=footer_command
        ).pack(anchor="w")

    def handle_login(self):
        username = self.auth_username_var.get().strip()
        password = self.auth_password_var.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Enter both username and password")
            return

        password_hash = self.hash_password(password)
        self.cursor.execute("SELECT username FROM users WHERE username=%s AND password_hash=%s", (username, password_hash))
        if self.cursor.fetchone() is None:
            messagebox.showerror("Error", "Invalid username or password")
            return

        self.current_user = username
        self.show_main_application()

    def handle_register(self):
        username = self.auth_username_var.get().strip()
        password = self.auth_password_var.get().strip()
        confirm_password = self.auth_confirm_var.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Enter username and password")
            return

        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match")
            return

        self.cursor.execute("SELECT 1 FROM users WHERE username=%s", (username,))
        if self.cursor.fetchone() is not None:
            messagebox.showerror("Error", "Username already exists")
            return

        try:
            self.cursor.execute(
                "INSERT INTO users(username, password_hash) VALUES(%s, %s)",
                (username, self.hash_password(password))
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            messagebox.showerror("Error", "Failed to create account")
            return

        messagebox.showinfo("Success", "Account created successfully! Please log in.")
        self.auth_username_var.set(username)
        self.auth_password_var.set("")
        self.auth_confirm_var.set("")
        self.switch_auth_mode("login")

    def show_main_application(self):
        if hasattr(self, "auth_container") and self.auth_container.winfo_exists():
            self.auth_container.destroy()
        self.build_ui()
        self.show_page("dashboard")

    def build_ui(self):
        self.sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", padding=(20, 24))
        self.sidebar.pack(side="left", fill="y")

        tk.Label(
            self.sidebar,
            text="FaceFlow",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg="#111827",
            anchor="w"
        ).pack(anchor="w", pady=(0, 28))

        tk.Label(
            self.sidebar,
            text="Attendance Control Center",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#111827",
            anchor="w"
        ).pack(anchor="w", pady=(0, 28))

        self.sidebar_buttons = {}
        for page_key, label in [
            ("dashboard", "Dashboard"),
            ("register", "Register Student"),
            ("train", "Train Model"),
            ("attendance", "Take Attendance"),
            ("reports", "Reports"),
        ]:
            button = tk.Button(
                self.sidebar,
                text=label,
                bg="#111827",
                fg="white",
                font=("Segoe UI", 11, "bold"),
                bd=0,
                pady=10,
                anchor="w",
                width=24,
                cursor="hand2",
                command=lambda key=page_key: self.show_page(key)
            )
            button.pack(fill="x", pady=4)
            self.sidebar_buttons[page_key] = button

        tk.Button(
            self.sidebar,
            text="Log Off",
            bg="#475569",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            pady=10,
            cursor="hand2",
            command=self.log_off
        ).pack(fill="x", pady=(12, 0))

        tk.Button(
            self.sidebar,
            text="Exit",
            bg="#dc2626",
            fg="#ffffff",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            pady=10,
            cursor="hand2",
            command=self.on_close
        ).pack(fill="x", pady=(24, 0))

        self.main = ttk.Frame(self.root, style="Content.TFrame")
        self.main.pack(side="left", fill="both", expand=True)

        self.header = ttk.Frame(self.main, style="Content.TFrame", padding=(24, 24, 24, 12))
        self.header.pack(fill="x")

        self.title_var = tk.StringVar()
        self.subtitle_var = tk.StringVar()

        tk.Label(
            self.header,
            textvariable=self.title_var,
            font=("Segoe UI", 24, "bold"),
            fg="#0f172a",
            bg="#f4f7fb",
            anchor="w"
        ).pack(anchor="w")

        tk.Label(
            self.header,
            textvariable=self.subtitle_var,
            font=("Segoe UI", 11),
            fg="#64748b",
            bg="#f4f7fb",
            anchor="w"
        ).pack(anchor="w", pady=(4, 0))

        self.content = ttk.Frame(self.main, style="Content.TFrame", padding=(24, 10, 24, 24))
        self.content.pack(fill="both", expand=True)

    def log_off(self):
        self.current_user = None

        if hasattr(self, "sidebar") and self.sidebar.winfo_exists():
            self.sidebar.destroy()
        if hasattr(self, "main") and self.main.winfo_exists():
            self.main.destroy()
        if hasattr(self, "auth_container") and self.auth_container.winfo_exists():
            self.auth_container.destroy()

        self.auth_mode = "login"
        self.build_auth_ui()
        self.auth_username_var.set("")
        self.auth_password_var.set("")
        self.auth_confirm_var.set("")

    def show_page(self, page_name):
        for child in self.content.winfo_children():
            child.destroy()

        for key, button in self.sidebar_buttons.items():
            button.configure(bg="#111827", fg="white")
        self.sidebar_buttons.get(page_name, self.sidebar_buttons["dashboard"]).configure(bg="#2563eb", fg="white")

        if page_name == "dashboard":
            self.set_header("System Overview", "Live attendance insights and quick actions")
            self.build_dashboard()
        elif page_name == "register":
            self.set_header("Register Student", "Capture a new face dataset and store student details")
            self.build_register_page()
        elif page_name == "train":
            self.set_header("Train Recognition Model", "Build the face recognition model from captured images")
            self.build_train_page()
        elif page_name == "attendance":
            self.set_header("Take Attendance", "Run live face recognition and mark attendance")
            self.build_attendance_page()
        elif page_name == "reports":
            self.set_header("Reports", "Generate attendance reports and export CSV data")
            self.build_reports_page()

    def set_header(self, title, subtitle):
        self.title_var.set(title)
        self.subtitle_var.set(subtitle)

    def build_dashboard(self):
        stat_cards = [
            ("Students", self.get_student_count(), "#2563eb"),
            ("Attendance Records", self.get_attendance_count(), "#0f766e"),
            ("Today Present", self.get_today_present_count(), "#7c3aed"),
        ]

        cards_frame = tk.Frame(self.content, bg="#f4f7fb")
        cards_frame.pack(fill="x")
        for index, (label, value, color) in enumerate(stat_cards):
            card = tk.Frame(cards_frame, bg="white", bd=1, relief="solid", padx=18, pady=18)
            card.grid(row=0, column=index, padx=(0, 14), pady=(0, 16), sticky="nsew")
            tk.Label(card, text=label, font=("Segoe UI", 11), fg="#64748b", bg="white").pack(anchor="w")
            tk.Label(card, text=str(value), font=("Segoe UI", 24, "bold"), fg=color, bg="white").pack(anchor="w", pady=(8, 0))
        cards_frame.columnconfigure(0, weight=1)
        cards_frame.columnconfigure(1, weight=1)
        cards_frame.columnconfigure(2, weight=1)

        main_panel = tk.Frame(self.content, bg="white", bd=1, relief="solid", padx=20, pady=20)
        main_panel.pack(fill="both", expand=True)

        tk.Label(main_panel, text="Quick Actions", font=("Segoe UI", 16, "bold"), bg="white", fg="#111827").pack(anchor="w")
        tk.Label(main_panel, text="Use the sidebar to switch between registration, training, recognition, and report modules.", font=("Segoe UI", 11), bg="white", fg="#64748b").pack(anchor="w", pady=(6, 16))

        action_frame = tk.Frame(main_panel, bg="white")
        action_frame.pack(fill="x")
        for page_key, label in [("register", "Register Student"), ("train", "Train Model"), ("attendance", "Take Attendance"), ("reports", "Generate Report")]:
            tk.Button(
                action_frame,
                text=label,
                bg="#2563eb",
                fg="white",
                font=("Segoe UI", 10, "bold"),
                bd=0,
                padx=12,
                pady=8,
                cursor="hand2",
                command=lambda key=page_key: self.show_page(key)
            ).pack(side="left", padx=(0, 10), pady=(0, 8))

        tk.Label(main_panel, text="Recent Attendance", font=("Segoe UI", 14, "bold"), bg="white", fg="#111827").pack(anchor="w", pady=(20, 8))
        recent_list = self.get_recent_records()
        if recent_list:
            for item in recent_list:
                tk.Label(main_panel, text=f"{item[1]} • {item[2]} • {item[3]}", font=("Segoe UI", 10), bg="white", fg="#334155").pack(anchor="w", pady=2)
        else:
            tk.Label(main_panel, text="No attendance records yet.", font=("Segoe UI", 10), bg="white", fg="#64748b").pack(anchor="w")

    def build_register_page(self):
        panel = tk.Frame(self.content, bg="white", bd=1, relief="solid", padx=24, pady=24)
        panel.pack(fill="both", expand=True)

        tk.Label(panel, text="Student Details", font=("Segoe UI", 16, "bold"), bg="white", fg="#111827").pack(anchor="w")
        tk.Label(panel, text="Enter the student ID and name, then capture 50 face samples from the webcam.", font=("Segoe UI", 11), bg="white", fg="#64748b").pack(anchor="w", pady=(6, 20))

        form = tk.Frame(panel, bg="white")
        form.pack(anchor="w")

        tk.Label(form, text="Student ID", font=("Segoe UI", 11, "bold"), bg="white", fg="#334155").grid(row=0, column=0, sticky="w", pady=8)
        self.id_var = tk.StringVar()
        tk.Entry(form, textvariable=self.id_var, font=("Segoe UI", 11), width=28).grid(row=0, column=1, padx=(12, 0), pady=8)

        tk.Label(form, text="Student Name", font=("Segoe UI", 11, "bold"), bg="white", fg="#334155").grid(row=1, column=0, sticky="w", pady=8)
        self.name_var = tk.StringVar()
        tk.Entry(form, textvariable=self.name_var, font=("Segoe UI", 11), width=28).grid(row=1, column=1, padx=(12, 0), pady=8)

        button_frame = tk.Frame(panel, bg="white")
        button_frame.pack(anchor="w", pady=(16, 0))

        tk.Button(
            button_frame,
            text="Capture Dataset",
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.register_student
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            button_frame,
            text="Remove Student",
            bg="#dc2626",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.remove_student
        ).pack(side="left")

    def build_train_page(self):
        panel = tk.Frame(self.content, bg="white", bd=1, relief="solid", padx=24, pady=24)
        panel.pack(fill="both", expand=True)

        tk.Label(panel, text="Train Model", font=("Segoe UI", 16, "bold"), bg="white", fg="#111827").pack(anchor="w")
        tk.Label(panel, text="The system will read captured images from the dataset folder and create the trainer model.", font=("Segoe UI", 11), bg="white", fg="#64748b").pack(anchor="w", pady=(6, 16))
        tk.Button(
            panel,
            text="Train Now",
            bg="#f59e0b",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.train_model
        ).pack(anchor="w")

    def build_attendance_page(self):
        panel = tk.Frame(self.content, bg="white", bd=1, relief="solid", padx=24, pady=24)
        panel.pack(fill="both", expand=True)

        tk.Label(panel, text="Live Recognition", font=("Segoe UI", 16, "bold"), bg="white", fg="#111827").pack(anchor="w")
        tk.Label(panel, text="Start the webcam and mark attendance automatically when a known face is detected.", font=("Segoe UI", 11), bg="white", fg="#64748b").pack(anchor="w", pady=(6, 16))
        tk.Button(
            panel,
            text="Start Recognition",
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.recognize_face
        ).pack(anchor="w")

    def build_reports_page(self):
        panel = tk.Frame(self.content, bg="white", bd=1, relief="solid", padx=24, pady=24)
        panel.pack(fill="both", expand=True)

        tk.Label(panel, text="Generate Report", font=("Segoe UI", 16, "bold"), bg="white", fg="#111827").pack(anchor="w")
        tk.Label(panel, text="Export a CSV file and generate charts from attendance records.", font=("Segoe UI", 11), bg="white", fg="#64748b").pack(anchor="w", pady=(6, 16))
        tk.Button(
            panel,
            text="Generate Monthly Report",
            bg="#7c3aed",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            command=self.generate_report
        ).pack(anchor="w")

    def register_student(self):
        sid = self.id_var.get().strip() if hasattr(self, "id_var") else ""
        name = self.name_var.get().strip() if hasattr(self, "name_var") else ""

        if sid == "" or name == "":
            messagebox.showerror("Error", "Enter Student ID and Name")
            return

        self.cursor.execute("SELECT 1 FROM students WHERE id=%s", (sid,))
        if self.cursor.fetchone() is not None:
            messagebox.showerror("Error", f"Student ID {sid} is already taken")
            return

        try:
            self.cursor.execute("INSERT INTO students(id, name) VALUES(%s, %s)", (sid, name))
            self.db.commit()
        except Exception:
            self.db.rollback()
            messagebox.showerror("Error", "Failed to register student")
            return

        self.initialize_facenet_components()
        cam = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, img = cam.read()
            if not ret:
                break

            faces = self.detect_faces(img)
            for (x, y, w, h) in faces:
                count += 1
                face_crop = img[y:y + h, x:x + w]
                filename = f"dataset/User.{sid}.{count}.jpg"
                cv2.imwrite(filename, face_crop)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"Images Captured: {count}/50", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Register Student", img)
            if cv2.waitKey(1) == 27 or count >= 50:
                break

        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", "Student registered successfully!")

    def remove_student(self):
        sid = self.id_var.get().strip() if hasattr(self, "id_var") else ""

        if sid == "":
            messagebox.showerror("Error", "Enter Student ID")
            return

        self.cursor.execute("SELECT name FROM students WHERE id=%s", (sid,))
        student = self.cursor.fetchone()
        if student is None:
            messagebox.showerror("Error", f"Student ID {sid} does not exist")
            return

        try:
            self.cursor.execute("DELETE FROM students WHERE id=%s", (sid,))
            self.db.commit()
        except Exception:
            self.db.rollback()
            messagebox.showerror("Error", "Failed to remove student")
            return

        removed_count = 0
        if os.path.isdir("dataset"):
            for filename in os.listdir("dataset"):
                if filename.startswith(f"User.{sid}.") and filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    os.remove(os.path.join("dataset", filename))
                    removed_count += 1

        messagebox.showinfo("Success", f"Student {sid} removed successfully. {removed_count} face image(s) deleted.")

    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, filename) for filename in os.listdir(path)]
        embeddings = []
        ids = []

        for image_path in image_paths:
            if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            student_id = self.parse_student_id_from_filename(os.path.basename(image_path))
            if student_id is None:
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            embedding = self.get_face_embedding(image, from_bgr=True)
            if embedding is None:
                continue

            embeddings.append(embedding)
            ids.append(student_id)

        return embeddings, ids

    def train_model(self):
        if not os.path.exists("dataset"):
            messagebox.showerror("Error", "Dataset folder not found!")
            return

        embeddings, ids = self.get_images_and_labels("dataset")

        if len(embeddings) == 0:
            messagebox.showerror("Error", "No face images found. Please register students first.")
            return

        os.makedirs("trainer", exist_ok=True)
        np.savez("trainer/facenet_embeddings.npz", embeddings=np.array(embeddings, dtype=np.float32), ids=np.array(ids, dtype=np.int32))
        messagebox.showinfo("Success", "FaceNet embeddings trained successfully!")

    # def mark_attendance(self, student_id, student_name):
    #     today = datetime.now().strftime("%Y-%m-%d")
    #     current_time = datetime.now().strftime("%H:%M:%S")

    #     self.cursor.execute("SELECT * FROM attendance WHERE id=%s AND date=%s", (student_id, today))
    #     record = self.cursor.fetchone()

    #     if record is None:
    #         self.cursor.execute(
    #             "INSERT INTO attendance(id, name, date, time, status) VALUES(%s, %s, %s, %s, %s)",
    #             (student_id, student_name, today, current_time, "Present"),
    #         )
    #         self.db.commit()

    def mark_attendance(self, student_id, student_name):
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
    
        self.cursor.execute(
            "SELECT * FROM attendance WHERE id=%s AND date=%s",
            (student_id, today)
        )
        record = self.cursor.fetchone()
    
        if record is None:
            self.cursor.execute(
                "INSERT INTO attendance(id, name, date, time, status) VALUES(%s, %s, %s, %s, %s)",
                (student_id, student_name, today, current_time, "Present")
            )
            self.db.commit()
    
            # Success Message
            messagebox.showinfo(
                "Attendance",
                f"Attendance taken successfully!\n\n"
                f"Student: {student_name}\n"
                f"Time: {current_time}"
            )
    
        else:
            messagebox.showinfo(
                "Attendance",
                f"{student_name}'s attendance is already marked today."
            )
    
        
    def recognize_face(self):
        if not os.path.exists("trainer/facenet_embeddings.npz"):
            messagebox.showerror("Error", "Train the FaceNet model first.")
            return

        self.initialize_facenet_components()
        training_data = np.load("trainer/facenet_embeddings.npz")
        stored_embeddings = training_data["embeddings"]
        stored_ids = training_data["ids"]
        cam = cv2.VideoCapture(0)

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            for (x, y, w, h) in faces:
                face_crop = frame[y:y + h, x:x + w]
                current_embedding = self.get_face_embedding(face_crop, from_bgr=True)
                if current_embedding is None:
                    continue

                best_match_id = None
                best_similarity = 0.0
                for embedding, student_id in zip(stored_embeddings, stored_ids):
                    similarity = self.cosine_similarity(current_embedding, embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = int(student_id)

                if best_match_id is not None and best_similarity >= 0.72:
                    self.cursor.execute("SELECT name FROM students WHERE id=%s", (best_match_id,))
                    result = self.cursor.fetchone()
                    if result:
                        name = result[0]
                        self.mark_attendance(best_match_id, name)
                        color = (0, 255, 0)
                        text = name
                    else:
                        color = (0, 0, 255)
                        text = "Unknown"
                else:
                    color = (0, 0, 255)
                    text = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition Attendance", frame)
            if cv2.waitKey(1) == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    def generate_report(self):
        query = "SELECT * FROM attendance"
        df = pd.read_sql_query(query, self.db)

        if df.empty:
            messagebox.showerror("Error", "No attendance records found.")
            return

        df.to_csv("attendance_report.csv", index=False)
        report = df.groupby("name").size().reset_index(name="Days Present")

        plt.figure(figsize=(8, 5))
        plt.bar(report["name"], report["Days Present"])
        plt.title("Monthly Attendance")
        plt.xlabel("Students")
        plt.ylabel("Days Present")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.pie(report["Days Present"], labels=report["name"], autopct="%1.1f%%", startangle=90)
        plt.title("Attendance Distribution")
        plt.show()

        heat = pd.crosstab(df["name"], df["date"])
        plt.figure(figsize=(12, 6))
        sns.heatmap(heat, cmap="YlGnBu", linewidths=0.5, annot=True)
        plt.title("Attendance Heatmap")
        plt.tight_layout()
        plt.show()

        messagebox.showinfo("Success", "Report generated successfully!\nCSV saved as attendance_report.csv")

    def get_student_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM students")
        return self.cursor.fetchone()[0]

    def get_attendance_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM attendance")
        return self.cursor.fetchone()[0]

    def get_today_present_count(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self.cursor.execute("SELECT COUNT(*) FROM attendance WHERE date=%s", (today,))
        return self.cursor.fetchone()[0]

    def get_recent_records(self):
        self.cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC LIMIT 6")
        return self.cursor.fetchall()

    def on_close(self):
        self.db.close()
        self.root.destroy()


if __name__ == "__main__":
    AttendanceApp()


