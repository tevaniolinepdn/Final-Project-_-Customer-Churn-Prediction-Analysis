# **Customer Churn Prediction Analysis**
Data Science perusahaan e-commerce "FUTURA"  melakukan prediksi terhadap customer akan  berhenti berlangganan atau tidak.
    ![](https://d35fo82fjcw0y8.cloudfront.net/2017/09/26225705/header%402x.png)
    
    
## **1. Background**

#### - **Problem Statement**
Di e-commerce FUTURA terdapat sekitar 16,8% dari total jumlah customer yang memilih berhenti berlangganan (churn). Perusahaan membutuhkan solusi untuk meminimalisir tingkat churn, karena diperkirakan biaya untuk mempertahankan customer yang sudah ada ada jauh lebih murah dibandingkan untuk mengakuisisi customer baru.      
#### - **Goal**
Decrease customer churn rate (Menurunkan customer churn rate)      
#### - **Objectives**
1. Membuat model machine learning untuk memprediksi customer yang berpotensi untuk churn
2. Menurunkan churn rate dari 16,8% menjadi 7%   
#### - **Business Metrics** 
Churn rate (% of customer that churn)
    
    
    
## **2. Strategy**
Berikut solusi untuk memprediksi customer yang akan berhenti berlangganan (churn) atau tidak menggunakan machine learning.
    
#### Step 1. Dataset: 
Download dataset melalui Kaggle, klik link [dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

#### Step 2. Exploratory Data Analysis (EDA) & Insights : 
Pada bagian ini dilakukan eksplorasi data seperti univariate analysis dan multivariate analysis untuk lebih memahami data di dalamnya. Pada step ini juga diperoleh gambaran dari business insight yang akan dieksplor lebih dalam lagi.

#### Step 3. Data Pre-processing & Preparation
Berikut langkah-langkah selama melakukan data pre-processing:
- **Handling missing value**: menggunakan median, metode ffill, dan if condition
- **Feature Engineering** : membuat feature baru diantaranya fitur WeekSinceLastOrder, OrderMean, CashbackRate, dan GetCoupon
- **Feature Selection** : memilih fitur-fitur terbaik dan melakukan drop atau filter feature yang tidak digunakan selama proses data seperti feature DaySinceLastOrder, OrderCount, CouponUsed, CustomerID
- **Feature Transformation** : Untuk menangani outliers digunakan Log Transformation dan Standardization
- **Feature Encoding** : Untuk mengubah categorical feature menjadi numeric feature. Feature encoding menggunakan One Hot Encoding.
- **Class Imbalance** : Karena kondisi data imbalance atau distribusi nilai unik pada target ini timpang, maka digunakan teknik SMOTE dan Oversampling. Oversampling dilakukan karena jumlah dataset sedikit.

#### Step 4. Modelling
Langkah dimana algoritma model machine learning dilatih dan melihat bagaimana model bisa melakukan prediksi data. Kemudian, cross validation digunakan untuk mengetahui hasil validasi dari proses pembelajaran model.Parameter yang diutamakan adalah nilai Recall, karena tidak ingin salah dalam memprediksi customer yang akan churn.

#### Step 5. Hyperparameter Tuning
Memilih model yang memiliki hasil paling baik untuk diaplikasikan ke project, dimana model perlu dituning kembali untuk menghasilkan score parameter yang lebih baik lagi.

#### Step 6. Feature Importance 
Hasil feature importance dapat digunakan untuk business insight maupun business recommendation pada tahap selanjutnya.

#### Step 7. Business Recommendation
Tahap ini terdapat hasil setelah dilakukan modelling dengan beberapa asumsi, kemudian business recommendation dari segala insight yang diperoleh pada tahap sebelumnya. 



## **3. Data Insights**
- **Customer Churn Based on Tenure**
Customer yang berhenti berlangganan (churn) kebanyakan memiliki masa berlangganan hanya di 2 bulan pertama saja.
![](https://github.com/tevaniolinepdn/Final-Project-_-Customer-Churn-Prediction-Analysis/blob/main/1.jpg?raw=true)

- **Customer Churn Based on Cashback**
Customer yang memperoleh cashback bonus dengan kategori medium (<200) lebih banyak yang memilih untuk berhenti berlangganan (churn)

![](https://raw.githubusercontent.com/tevaniolinepdn/Image-Insight/3cc250502caeb553c8b4e3588defb2663bcfb208/2%20-%20average%20tenure%20based%20on%20cashback%20category.jpg?token=A3JF6OHD6QKMGDBXKUFAPOTDRG6AC)

![](https://raw.githubusercontent.com/tevaniolinepdn/Image-Insight/3cc250502caeb553c8b4e3588defb2663bcfb208/3%20-%20Customers%20with%20lower%20cashback%20are%20more%20likely%20to%20churn.jpg?token=A3JF6OGT3OPKH3XHOOBGKJ3DRG6AC)


- **Customer Churn Based on Complain**
Menunjukkan bahwa customer yang mengajukan complain lalu berhenti berlangganan (churn) jumlahnya lebih banyak 31%.

![](https://raw.githubusercontent.com/tevaniolinepdn/Image-Insight/3cc250502caeb553c8b4e3588defb2663bcfb208/4%20-%20Customer%20churn%20based%20on%20complain.jpg?token=A3JF6OANYQLXZK5EZVZVFSTDRG6AC)

- **Customer Churn Based on Order Category**
Kategori order customer yang paling banyak mengajukan komplain adalah jenis **"Grocery"**, yang selanjutnya diikuti oleh jumlah komplain customer yang pernah membeli jenis **"Mobile phone"** serta **"Fashion"**.

![](https://raw.githubusercontent.com/tevaniolinepdn/Image-Insight/3cc250502caeb553c8b4e3588defb2663bcfb208/6%20-%20Complain%20rate%20based%20on%20order%20category.jpg?token=A3JF6OATJU3HFTTHOKO6PALDRG6AC)

- **Churn Rate Based on Order Category**
Apabila melihat dari jumlah churn rate berdasarkan Order Category ternyata customer yang pernah membeli jenis **"Mobile Phone"** adalah yang paling banyak churn mencapai 27.4%. 

![](https://raw.githubusercontent.com/tevaniolinepdn/Image-Insight/3cc250502caeb553c8b4e3588defb2663bcfb208/5%20-%20Churn%20rate%20based%20on%20order%20category.jpg?token=A3JF6OFCWVKJBTOGQ3DEB4LDRG6AC)


## **4. Machine Learning Model**
- Rasio Train-Test Split 4:1
- Class Imbalance menggunakan SMOTE method (1 : 0,5) atau Class 0 : Class 1 = 4519 : 2259

- Pada tahap ini mencoba beberapa model seperti Logistic Regression, KNN, Decision Tree, Random Forest, Adaboost, XGBOOST, dan Catboost. 
- Metrics yang digunakan yaitu **Recall** atau Negative False. Karena tidak ingin customer yang akan churn terdeteksi tidak churn, padalah aktualnya dia akan churn. Upayanya kita akan melakukan treatment kepada customer yang akan churn tersebut. 
- Model yang dipiih sebagai model paling ideal dari beberapa eksprerimen adalah model **"Catboost+ Hyperparameter Tuning"**. 


| Accuracy Train | Accuracy Tes | Precision Train | Precision Test  | Recall Train | Recall Test | F1 Score Train | F1 Score Test  | CrossVal Recall Train | Crossval Recall Test
| :----------- | :------------- | :----------- | :------- | :------------- | :------------ | :------------- | :----------- | :---------- | :------------: | 
|  0,899608  | 0,893238 | 0,873531 | 0,839535  | 0,822012  |  0,816742  | 0,846989  | 0,827982|  0,819947 | 0,800063 |
||

- Model Catboost + Hyperparameter Tuning memiliki nilai recall train : 0,822012 dan recal test : 0,816742. Kemudian, cross validation train : 0,819947 dan crossvalidation test : 0,800063  


**Confusion Matrix**
- Class 0 = 904 
- Class 1 = 452
- Recall test = 0,8167

| | Predictive Class Positive | Predicted Class Negative 
|------------ | :------- | :-------- 
|Actual Class Positive | 369 |83 | :----------------- | :------------ | :------------| 
|Actual Class Negative | 70 | 834 | :---------------- | :------------- |:---------------- |
||


## **5. Feature Importance**
![](https://raw.githubusercontent.com/tevaniolinepdn/Image-Insight/3cc250502caeb553c8b4e3588defb2663bcfb208/Feature%20importance.jpg?token=A3JF6OGQ6CIEALGE6OGTQWLDRG6AC)

- Feature yang paling penting berdasarkan shap value adalah **Tenure** dan **Complain**. 




## **6. Decreasing Churn Rate Simulation**
- Percentage of Churn = 16,84%
- Amount of Churn = 948 customers

**AFTER MODELLING**
- Recall 80%
- 7,1 % dari Recall, customer yang kemungkinan prediksinya 0,95 sampai 1 
- Asumsi 10% dari Recall, customer yang churn mendapatkan serta menggunakan services/treatment
- Asumsi 10% dari recall, misshandling

1.  80 - (7,1% * 80) - (10% * 80) = 66,32%
2.  80 - (7,1% * 80) - (10% * 80) - (10% * 80) = 58,32%

#### Churn Rate
Churn rate dari asumsi sebelumnya:
1. 16,84 x (100% - 66,32%) = 5,66%
2. 16,84 x (100% - 58,32%) = 7,01%




## **7. Business Recommendation**
Setelah model machine learning telah diuji coba, selanjutnya terdapat rekomendasi bisnis yang sekiranya akan diaplikasikan kepada customer untuk upaya mengurangi churn/jumlah berhenti berlangganan.
Berikut upaya-upaya rekomendasi yang bisa diberikan kepada customer :

1. **Cross Selling**
Sebelumnya kita melihat bahwa banyak customer yang memiliki tenure singkat sekitar 2 bulan pertama namun sudah berhenti berlangganan. Maka, untuk memperpanjang tenure customer, kita bisa mencoba untuk melakukan cross selling misalnya customer yang telah melakukan pembelian kategori **Laptop & Accessory** bisa dimunculkan halaman seperti mouse, keyboards, atau paket pembersih perangkat barang elektronik. Berdasarkan report 2006 (Forbes), Amazon meng-klaim bahwa 35% dari pendapatannya didorong dari cross-sales. ([Source 1](https://uploads-ssl.webflow.com/5f7da44a7aa7a96256f38ff8/609517d1fda37d08c9acfee2_The%20Importance%20of%20Cross-Selling%20in%20E-Commerce.pdf))

2. **Display the Original Product**
Menyarankan kepada para penjual di e-commerce untuk mencantumkan foto produk yang asli, jelas, dan tetap menarik. Selain itu, disarankan untuk melengkapi deksripsi produk yang lengkap dan jelas serta himbauan agar customer selalu membaca deksripsi produk sebelumn membeli. Upaya ini dilakukan untuk menghindari complain dari customer dan memutuskan berhenti berlangganan. Customer bisa saja mengajukan complain dan churn karena merasa dibohongi seperti produk yang diterima tidak sesuai. 
([Source 2](https://www.mas-software.com/blog/churn-rate-cara-mengurangi))
([Source 3](https://www.marketingcharts.com/wp-content/uploads/2018/05/Namagoo-Elements-Great-Online-Shopping-Experience-May2018.png))


3. **Seasonal Promo and Reward**
Sebagai upaya untuk memperpanjang tenure customer bisa diberikan free delivery atau bonus cashback yang mana berdasarkan research by Dublin Business School dan artikel Harvard Business Review akan memberikan manfaat yang nyata bagi customeer dan diharapkan customer akan bertahan.

4. **Meningkatkan Service**
- Untuk mengurangi complain dari customer maka bisa disediakan customer service yang fast respond dan menangani komplain atau pertanyaan yang diajukan customer. 
- Untuk kategori **Grocery** pengirimannya harus lebih cepat dan tepat karena untuk memastikan kondisi barang seperti buah atau sayur masih bagus ketika sampai di tangan customer. ([Source 4](https://www.meteorspace.com/2022/08/25/statistics-that-prove-how-your-delivery-speed-impacts-your-business/))































     
     
     
     
    
    
    
      
    


