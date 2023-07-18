import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
import datetime
from datetime import date, time

st.title('Credit Card Transaction Checking')
st.write('''This model has some values below: 
            
    1. Accuration : 
    2. Recall : 
    3. Precission : 
    4. F1-Score : ''')



#interface
cc = st.text_input("Credit card number: ") #cc_num

option = st.selectbox(
    'Category: ',
    ("home", "kids_pets", "shopping_pos", "gas_transport", "food_dining", "personal_care", "health_fitness", "shopping_net", 
 "grocery_pos", "entertainment", "misc_pos", "misc_net", "travel", "grocery_net")) #category

amount = st.number_input("Amount (USD): ") #amt

notif = gen = ''
month = hour = pop = age = cek = 0

cek_cc = st.button("Check")
if cek_cc:
    df = pd.read_csv('https://drive.google.com/u/4/uc?id=1pAsuE9kKim-UbUCJEl-9DZJRJm6hviRm&export=download', parse_dates=['dob']) #, parse_dates=['dob']
    cc = int(cc)
    for i in range(len(df)):
        if cc == df['cc_num'][i]:
            now = datetime.datetime.now()
            current_date = now.date()
            month = now.month #trans_month
            hour = now.hour #trans_hour
            gen = df['gender'][i] #gender
            pop = df['city_pop'][i] #city_pop
            dob = df['dob'][i].date()
            age = current_date - dob
            age = int(np.floor(age.days/365)) #age

            data = {'cc_num': cc, 'trans_month': month, 'trans_hour': hour, 'gender': gen, 'city_pop': pop,
                    'age': age, 'category': option, 'amt': amount}
            X_test = pd.DataFrame([data])
            X_test.to_csv('data/ThisUserData.csv', index=False)

            cek = 1
            notif = 'Registered Card Number'
            break

    if cek == 0:
        data = {'cc_num': 0, 'trans_month': 0, 'trans_hour': 0, 'gender': '', 'city_pop': 0,
                'age': 0, 'category': '', 'amt': 0}
        X_test = pd.DataFrame([data])
        X_test.to_csv('data/ThisUserData.csv', index=False)
        notif = 'Unregistered Card Number!!!'


    st.write(notif)


deteksi = st.button("Detection")
if deteksi:
    test = pd.read_csv('data/ThisUserData.csv')
    database = pd.read_csv('data/database.csv')
    if test['cc_num'][0]==0:
        test['predict'] = 'Error'
        st.success('Transaksi Error')
        st.table(test)
        
    else:
        loaded_model = pickle.load(open('tools/Prepocessor.sav', 'rb'))
        X_test = loaded_model.transform(test)

        X_test = torch.FloatTensor(X_test.toarray())
        model = nn.Sequential(
            nn.Linear(56, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
    
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
    
            nn.Linear(8, 2),
            nn.LogSoftmax(1)
        )

        weights = torch.load("tools/smote+outlier_v1-mn42baru_weights.pth", map_location="cpu")
        model.load_state_dict(weights)

        with torch.no_grad():
            model.eval()
            output = model(X_test)
            y_pred = output.argmax(1)

        predict = ''
        if y_pred==0:
            test['predict'] = 'Normal'
            predict = 'Transaksi Normal'
        elif y_pred==1:
            test['predict'] = 'Penipuan'
            predict = 'Transaksi Penipuan'
        else:
            test['predict'] = 'Error'
            predict = 'Transaksi Error'

        database = pd.concat([database, test])
        database.to_csv('data/database.csv', index=False)
        st.success(predict)
        st.table(test)
