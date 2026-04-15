# Multi-Class News Topic Classification on Bitcoin Articles Using LSTM-Based Deep Learning

## 📌 Overview
Proyek ini bertujuan untuk mengklasifikasikan artikel berita terkait Bitcoin ke dalam berbagai kategori topik menggunakan pendekatan **Natural Language Processing (NLP)** dan **Deep Learning berbasis LSTM (Long Short-Term Memory)**.

Model dibangun menggunakan **TensorFlow/Keras** dan dilatih pada dataset artikel berita Bitcoin untuk menghasilkan klasifikasi multi-kelas secara otomatis berdasarkan isi teks (summary).
---

## 🎯 Objectives
* Mengolah data teks berita Bitcoin
* Melakukan preprocessing teks menggunakan teknik NLP
* Membangun model klasifikasi multi-kelas berbasis LSTM
* Mengevaluasi performa model menggunakan accuracy dan loss
---

## 📂 Dataset
Dataset yang digunakan berasal dari Kaggle:
🔗 [https://www.kaggle.com/datasets/balabaskar/bitcoin-news-articles-text-corpora](https://www.kaggle.com/datasets/balabaskar/bitcoin-news-articles-text-corpora)

Dataset berisi artikel berita Bitcoin dengan berbagai atribut seperti:
* summary (digunakan sebagai fitur utama)
* topic (label target)
---

## ⚙️ Data Preprocessing
Tahapan preprocessing yang dilakukan:

### 1. Data Cleaning
Menghapus kolom yang tidak relevan:

```python
['author', 'title', 'authors', 'country', 'language', 'excerpt',
 'article_id', 'published_date', 'link', 'clean_url', 'rights',
 'article_rank', 'media', 'twitter_account', 'article_score']
```

### 2. Encoding Label
Menggunakan **One-Hot Encoding** untuk mengubah label kategori:

```python
pd.get_dummies(df.topic)
```

Total kelas: **13 kategori**
* beauty
* business
* economics
* energy
* entertainment
* finance
* food
* news
* politics
* science
* sport
* tech
* world

---

### 3. Text Preprocessing (NLP)
Menggunakan library **NLTK**:
* Tokenization
* Stopword removal
* Cleaning text (menghapus karakter non-alfanumerik)

```python
tokenizer = nltk.RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words('english'))
```
---

### 4. Data Splitting
Dataset dibagi menjadi:
* 80% training
* 20% testing

```python
train_test_split(summary, label, test_size=0.2)
```
---

### 5. Tokenization & Padding
Menggunakan Keras:

```python
Tokenizer(num_words=5000, oov_token='')
pad_sequences()
```

---

## 🧠 Model Architecture
Model dibangun menggunakan **Sequential API**:

```python
Embedding(input_dim=5000, output_dim=100)
LSTM(64)
Dropout(0.5)
Dense(128, activation='relu')
Dense(526, activation='relu')
Dense(256, activation='relu')
Dense(13, activation='softmax')
```

### 🔍 Penjelasan Layer:
* **Embedding** → Representasi numerik kata
* **LSTM** → Menangkap konteks urutan teks
* **Dropout** → Mengurangi overfitting
* **Dense Layers** → Feature extraction lanjutan
* **Softmax Output** → Klasifikasi multi-kelas (13 kategori)

---

## ⚡ Training Configuration

```python
loss = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
epochs = 50
batch_size = 32
```

### 🎯 Custom Callback

Training akan berhenti otomatis jika:

```python
val_accuracy > 0.81
```

---

## 📊 Model Evaluation

### 1. Accuracy Graph
Menampilkan perbandingan:
* Training Accuracy
* Validation Accuracy

### 2. Loss Graph
Menampilkan:
* Training Loss
* Validation Loss

Visualisasi menggunakan:
```python
matplotlib.pyplot
```

---

## 📈 Results
Model mampu mencapai:
* Akurasi validasi > **81%**
* Training dihentikan otomatis saat target tercapai
---

## 🛠️ Technologies Used
* Python
* TensorFlow / Keras
* Pandas
* Numpy
* Scikit-learn
* NLTK
* Matplotlib
---

## 📦 Requirements
Buat file `requirements.txt` seperti berikut:

```
tensorflow
pandas
numpy
scikit-learn
nltk
matplotlib
```
---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('popular')
```

### 4. Run Notebook / Script

Jalankan di:

* Google Colab
* Jupyter Notebook
* Python environment lokal

---

## 📌 Notes

* Pastikan dataset sudah di-load dengan path yang benar:

```python
df = pd.read_csv('/content/bitcoin_articles.csv')
```

* Jika dijalankan secara lokal, sesuaikan path dataset

---

