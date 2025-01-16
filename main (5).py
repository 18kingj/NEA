import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm.decl_api import DeclarativeBase
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError
from enum import global_enum
import customtkinter as ctk


root = ctk.CTk()
root.geometry("400x400")
root.title("CustomTkinter")

data = pd.read_csv('loan-train.csv')
data.dropna(inplace=True)
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y_eligibility = data['Loan_Status'] 
y_loan_amount = data['LoanAmount']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
X_train, X_test, y_train_elig, y_test_elig = train_test_split(X_lstm, y_eligibility, test_size=0.2, random_state=42)
_, _, y_train_amount, y_test_amount = train_test_split(X_lstm, y_loan_amount, test_size=0.2, random_state=42)



class LoanPredictor():
    def __init__(self):
        self.Scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.model = None
        self.scaled_data = None
        self.X_train = None
        self.y_train = None

    def GetData(self, file_path):
        try:
            with open(file_path, "r") as file:
                self.data = file.read()
                self.data = self.data.split("\n")
                self.data = [self.data [4:8] + self.data[11:13]]
        except FileNotFoundError:
            print("File not found.")
    def PreprocessData(self):
        try:
            self.data = np.array(self.data, dtype=np.float32)
            self.scaled_data = self.Scaler.fit_transform(self.data)
            self.X_train, self.y_train = self.scaled_data[:, :-1], self.scaled_data[:, -1]
            self.X_train = self.X_train.reshape(-1, 1)
        except ValueError:
            print("Invalid data format.")
            
    def build_model(self):
        self.model = model = Sequential([
            LSTM(64, input_shape=(1, X_lstm.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid', name='eligibility')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        print('Model built succesfully! ')
        output_loan = Dense(1, activation='linear', name='loan_amount')
        self.model.add(output_loan)
        self.model.compile(optimizer='adam',
              loss={'eligibility': 'binary_crossentropy', 'loan_amount': 'mse'},
              metrics={'eligibility': 'accuracy', 'loan_amount': 'mae'})
        history = self.model.fit(X_train, {'eligibility': y_train_elig, 'loan_amount': y_train_amount},
            validation_data=(X_test, {'eligibility': y_test_elig, 'loan_amount': y_test_amount}),
            epochs=20, batch_size=32)
        model.save('loan_eligibility_lstm.h5')

    def train_model(self, epochs: int = 25, batch_size: int = 32):
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
        return history
    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float32)
        X_test = X_test.reshape(-1, 1)
        X_test = self.Scaler.transform(X_test)
        y_pred = self.model.predict(X_test)
        y_pred = self.Scaler.inverse_transform(y_pred)
        return y_pred[0][0]
        
    def plot_predictions(self, evaluation_results, save_path='prediction_plot.png'):
        plt.figure(figsize=(12, 6))
        plt.plot(evaluation_results['actual_values'],
                label='Actual Prices', color='blue')
        plt.plot(evaluation_results['predictions'],
                label='Predicted Prices', color='red')
        plt.title('Loan Eligibility Prediction Results')
        plt.xlabel('Time (days)')
        plt.ylabel('Loan eligibility')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
 
        username_entry = ctk.CTkEntry(root, placeholder_text = 'Enter your username: ', show='*')
        username_entry.place(relx=0.5, rely=0.1, anchor=ctk.CENTER)
        password_entry = ctk.CTkEntry(root, placeholder_text = 'Enter your password: ', show='*')
        password_entry.place(relx=0.5, rely=0.2, anchor=ctk.CENTER)
        show_password = ctk.BooleanVar()
        show_password_check = ctk.CTkCheckBox(root, text="Show password", variable=show_password, command=lambda: password_entry.configure(show="" if show_password.get() else "*"))
        show_password_check.place(relx=0.5, rely=0.3, anchor=ctk.CENTER)

    
        
def getfactors():
    ApplicantIncome_entry = ctk.CTkEntry(root, placeholder_text="Enter your income: ")
    ApplicantIncome_entry.pack(padx=20, pady=20)
    ApplicantIncome = ApplicantIncome_entry.get()
    if ApplicantIncome.isdigit():
        ApplicantIncome = int(ApplicantIncome)
                
    dependents_entry = ctk.CTkEntry(root, placeholder_text="Enter your number of dependents: ")
    dependents_entry.pack(padx=20, pady=20)
    dependents = dependents_entry.get()
    dependents_entry.grid(row=2, column=1, padx=10, pady=5)
        
    credit_history_entry = ctk.CTkEntry(root, placeholder_text="Enter your credit history: ")
    credit_history_entry.pack(padx=20, pady=20)
    credit_history = credit_history_entry.get()
    credit_history_entry.grid(row=8, column=1, padx=10, pady=5)
    

    education_entry = ctk.CTkEntry(root, placeholder_text="Enter your education(Graduate/ Not Graduate): ")
    education_entry.pack(padx=20, pady=20)
    education = education_entry.get()
    education_entry.grid(row=3, column=1, padx=10, pady=5)


    self_employed_entry = ctk.CTkEntry(root, placeholder_text="Enter if you are self employed")
    self_employed_entry.pack(padx=20, pady=20)
    self_employed = self_employed_entry.get()
    self_employed_entry.grid(row=4, column=1, padx=10, pady=5)

    return ApplicantIncome, dependents, credit_history, education, self_employed

engine = create_engine('sqlite:///User_and_details.db')
Session = sessionmaker(bind=engine)
session = Session()
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "user"
    UserID = Column(Integer, primary_key=True)
    Email = Column(String, unique=True)
    Password = Column(String)
    
    def get_user_id(self):
        return self.UserID
        
    def __init__(self, email, password):
        self.Email = email
        self.Password = password

    @staticmethod
    def get_user_by_email(Email: str):
        try:
            user = session.query(User).filter_by(Email=Email).first()
            return user
        except SQLAlchemyError as e:
            print(f"Error: {str(e)}")
            return None
            
    @staticmethod
    def get_user_by_id(UserID: int):
        try:
            user = session.query(User).filter_by(UserID=UserID).first()
            return user
        except SQLAlchemyError as e:
            print(f"Error: {str(e)}")
            return None
model = load_model('loan_eligibility_lstm.h5')
class Factors(Base):
    __tablename__ = "factors"
    Loan_ID = Column(String, primary_key=True)
    ApplicantIncome = Column(Integer)
    Dependents = Column(Integer)
    Credit_history = Column(Integer)
    Education = Column(String)
    Self_employed = Column(String)
    UserID = Column(Integer, ForeignKey("user.UserID"))
    
    def __init__(self, ApplicantIncome, dependents, credit_history, education, self_employed, user_id):
        self.ApplicantIncome = ApplicantIncome
        self.Dependents = dependents
        self.Credit_history = credit_history
        self.Education = education
        self.Self_employed = self_employed
        self.UserID = user_id
    
    global Edit_factors
    def Edit_factors(self, ApplicantIncome, dependents, credit_history, education, self_employed):
        print('Edit factors')
        print('applicant income: ', ApplicantIncome)
        print('dependents: ', dependents)
        print('credit history: ', credit_history)
        print('education:' , education)
        print('self employed: ', self_employed)
        choice_entry = ctk.CTkEntry(root, placeholder_text="Enter your choice to edit: ")
        choice_entry.pack(padx=20, pady=20)
        choice = choice_entry.get().lower()
        if choice == "applicant income":
            new_ApplicantIncome_entry = ctk.CTkEntry(root, placeholder_text="Enter your new income: ")
            new_ApplicantIncome_entry.pack(padx=20, pady=20)
            new_ApplicantIncome = new_ApplicantIncome_entry.get()
            if new_ApplicantIncome.isdigit():
                self.Income = int(new_ApplicantIncome)
                session.commit()
                print("Income updated successfully")
            else:
                print("Invalid input")
        elif choice == "dependents":
            new_dependents_entry = ctk.CTkEntry(root, placeholder_text="Enter your new dependents: ")
            new_dependents_entry.pack(padx=20, pady=20)
            new_dependents = new_dependents_entry.get()
            if new_dependents.isdigit():
                self.Dependents = int(new_dependents)
                session.commit()
                print("dependents updated successfully")
            else:
                print("Invalid input")
        elif choice == "credit history":
            new_credit_history_entry = ctk.CTkEntry(root, placeholder_text="Enter your new credit history: ")
            new_credit_history_entry.pack(padx=20, pady=20)
            new_credit_history = new_credit_history_entry.get()
            self.Credit_history = new_credit_history
            session.commit()
            print("Credit history updated successfully")
        elif choice == "education":
            new_education_entry = ctk.CTkEntry(root, placeholder_text="Enter your education status(Graduate/ Not Graduate): ")
            new_education_entry.pack(padx=20, pady=20)
            new_education = new_education_entry.get()
            if new_education.isdigit():
                self.Education = int(new_education)
                session.commit()
                print("Education updated successfully")
            else:
                print("Invalid input")
        elif choice == "self employed":
            new_self_employed_entry = ctk.CTkEntry(root, placeholder_text="Enter if you are self employed: ")
            new_self_employed_entry.pack(padx=20, pady=20)
            new_self_employed = new_self_employed_entry.get()
            self.self_employed = new_self_employed
            session.commit()
            print("self employment updated successfully")
        else:
            print("Invalid input. Try again")
            Edit_factors(self, ApplicantIncome, dependents, credit_history, education, self_employed)
        
def evaluate_model(self):
    test_start = len(self.data) - 30
    test_data = self.scaled_data[test_start - self.prediction_days:]

    x_test = []
    y_test = self.scaled_data[test_start:].reshape(-1)

    for x in range(self.prediction_days, len(test_data)):
        x_test.append(test_data[x-self.prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = self.model.predict(x_test)
    predictions = self.scaler.inverse_transform(predictions)
    actual_values = self.scaler.inverse_transform([y_test])

    mse = np.mean((predictions - actual_values.T) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actual_values.T))

root.mainloop()
