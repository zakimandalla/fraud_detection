import streamlit as st
import pandas as pd

st.title("Database Transaksi")
lihat = st.button("Lihat Database")
if lihat:
    tabel = pd.read_csv('data/database.csv')
    st.table(tabel)