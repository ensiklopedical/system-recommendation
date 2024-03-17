# Laporan Proyek Machine Learning - Faisal Ahmad Gifari

## Project Overview

Pada bagian ini, Kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa proyek ini penting untuk diselesaikan.
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
  
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

## Business Understanding

Pada bagian ini, Anda perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Approach” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution approach (algoritma atau pendekatan sistem rekomendasi).

## Data Understanding
Dataset yang digunakan untuk pembuatan model system recommendation ini adalah dataset "MovieLens Latest" yang tersedia di situs [grouplens](https://grouplens.org/datasets/movielens/) yang berisi data-data mengenai film-film beserta rating yang diberikan oleh para pengguna. Dataset ini terakhir di-update pada September 2018

Terdapat banyak file didalamnya, tetapi yang digunakan hanya dataset `movie.csv` dan `rating.csv`. `movie.csv` terdiri dari 9078 baris data dan 3 kolom data. Kemudian, `rating.csv` terdiri dari 100836 baris data dan 4 kolom data.

Kedua dataset tersebut dapat digunakan untuk membuat system recommendation, baik `Content-Based Filtering` maupun `Collaborative Filtering`

Dataset tersebut dapat diunduh [disini](https://grouplens.org/datasets/movielens/)

[Direct download](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

Berikut ini adalah infomasi lainnya mengenai atribut-atribut yang terdapat pada dua dataset tersebut:


Atribut-atribut pada `movie.csv`:
- ```movieId```: Id film
- ```title```: Judul film
- ```genres```: Genre film


Atribut-atribut pada `rating.csv` :
- ```userId```: Id user
- ```movieId```: Id film
- ```rating```: Skor rating yang sebuah film
- ```timestamp```: Waktu kapan film diberikan skor rating



Dataset `movie.csv` ditampung dalam variabel `movie_df`

Dataset `rating.csv` ditampung dalam variabel `review_df`

```python

```

**_Exploratory Data Analysis_**

Exploratory Data Analysis (EDA) adalah pendekatan analisis data yang bertujuan untuk memahami karakteristik utama dari kumpulan data. EDA melibatkan penggunaan teknik statistik dan visualisasi grafis untuk menemukan pola, hubungan, atau anomali untuk membentuk hipotesis. Proses ini sering kali tidak terstruktur dan dianggap sebagai langkah awal penting dalam analisis data yang membantu menentukan arah analisis lebih lanjut.

Berikut ini adalah EDA yang dilakukan untuk `movie_df`:

- ```python
  movie_df.shape
  ```
  Kode diatas memiliki output:
  ```python
  (9742, 3)
  ```
  Berdasarkan output diatas, `movie_df` memiliki:
  - 9742 baris data
  - 3 kolom data
  
- ```python
  movie_df.keys()
  ```
  Kode diatas memiliki output:
  ```python
  Index(['movieId', 'title', 'genres'], dtype='object')
  ```
  Berdasarkan output diatas, `movie_df` memiliki 3 kolom berbeda, yaitu:
  - `movieId`
  - `title`
  - `genres`
  
- ```python
  movie_df.info()
  ```
  Kode diatas memiliki output:
  ```python
  RangeIndex: 9742 entries, 0 to 9741
  Data columns (total 3 columns):
   #   Column   Non-Null Count  Dtype 
  ---  ------   --------------  ----- 
   0   movieId  9742 non-null   int64 
   1   title    9742 non-null   object
   2   genres   9742 non-null   object
  dtypes: int64(1), object(2)
  ```
  Berdasarkan output diatas, `movie_df` memiliki 3 kolom tersebut memiliki tipe datanya masing-masing, yaitu:
  - `movieId` = `int64`
  - `title` = `object`
  - `genres` = `object`
  
  
- ```python
  print(movie_df['movieId'].nunique())
  ```
  Kode diatas memiliki output:
  ```python
  9742
  ```
  Berdasarkan output diatas, movie_df memiliki terlalu banyak genre nyaris untuk setiap film-nya. Hal ini perlu disimplifikasikan untuk memudahkan tahap-tahap lainnya.
  
- ```python
  movie_df['genres'] = movie_df['genres'].str.split('|').str[0]
  ```
  Kode diatas membuat kolom 'genre' hanya mempertahankan genre yang berada padaa urutan pertama saja. Hal ini mempermudah untuk pemrosesan dataset pada tahap-tahap selanjutnya. Proses ini dilakukan lebih awal untuk memudahkan proses selanjutnya, khususnya agar dapat dilakukannya Visualisasi Data. 
  
- ```python
  unique_genres = movie_df['genres'].unique()
  for genre in unique_genres:
      print(genre)
  ```
  Kode diatas memiliki output:
  ```python
  Adventure
  Comedy
  Action
  Drama
  Crime
  Children
  Mystery
  Animation
  Documentary
  Thriller
  Horror
  Fantasy
  Western
  Film-Noir
  Romance
  Sci-Fi
  Musical
  War
  (no genres listed)
  ```
  Proses perubahan nilai pada kolom `genre` sudah berhasil dilakukan.


- ```python
  movie_df.rename(columns={'genres': 'genre'}, inplace=True)
  ```
  Nama kolom `genres` berhasil diganti menjadi `genre` karena genre dari tiap film sudah tidak ada yang lebih dari 1. Proses ini dilakukan lebih awal untuk memudahkan proses selanjutnya
  
Masih ada beberapa tindakan yang perlu dilakukan untuk `movie_df`. Proses pembersihan dan persiapan dataset akan dikerjakan lebih lanjut pada tahap selanjutnya.

Berikut ini adalah EDA yang dilakukan untuk `review_df`:

- ```python
  review_df.shape
  ```
  Kode diatas memiliki iutput:
  ```python
  (100836, 4)
  ```
  Berdasarkan output diatas, `review_df` memiliki:
  - 100836 baris data
  - 4 kolom data

- ```python
  review_df.keys()
  ```
  Kode diatas memiliki iutput:
  ```python
  Index(['userId', 'movieId', 'rating', 'timestamp'], dtype='object')
  ```
  Berdasarkan output diatas, `review_df` memiliki 4 kolom berbeda, yaitu:
  - `userId`
  - `movieId`
  - `review`
  - `timestamp`

- ```python
  review_df.rename(columns={'rating': 'review'}, inplace=True)
  ```
  Tahap diatas adalah pengubahan nama kolom yang sebelumnya `review` menjadi `review` untuk memudahkan karena sesuai dengan nama dataframe-nya yaitu `review_df`.
  Proses prengubahan ini dilakukan lebih awal untuk memudahkan proses selanjutnya.

  ```python
  review_df.keys()
  ```
  Kode diatas memiliki output:
  ```python
  Index(['userId', 'movieId', 'review', 'timestamp'], dtype='object')
  ```
  Berdasarkan output diatas, proses pengubahan nama kolom `rating` menjadi `review` telah berhasil dilakukan


- ```python
  review_df.info()
  ```
  Kode diatas memiliki output:
  ```python
  Data columns (total 4 columns):
   #   Column     Non-Null Count   Dtype  
  ---  ------     --------------   -----  
   0   userId     100836 non-null  int64  
   1   movieId    100836 non-null  int64  
   2   review     100836 non-null  float64
   3   timestamp  100836 non-null  int64  
  dtypes: float64(1), int64(3)
  ```
  Berdasarkan output diatas, `review_df` memiliki 4 kolom berbeda dengan tipe datanya masing-masing, yaitu:
  - `userId` = `int64`
  - `movieId` = `int64`
  - `review` = `float64`
  - `timestamp` = `int64`

- ```python
  review_df['review'].describe()
  ```
  Kode diatas memiliki iutput:
  ```python
  count    100836.000000
  mean          3.501557
  std           1.042529
  min           0.500000
  25%           3.000000
  50%           3.500000
  75%           4.000000
  max           5.000000
  Name: review, dtype: float64
  ```
  Fungsi diatas memberikan informasi statistika deskriptif untuk kolom `review`, yaitu:
    - ```count``` : Jumlah data dari sebuah kolom
    - ```mean``` : Rata-rata dari sebuah kolom
    - ```std``` : Standar deviasi dari sebuah kolom
    - ```min``` : Nilai terendah pada sebuah kolom
    - ```25%``` : Nilai kuartil pertama (Q1) dari sebuah kolom
    - ```50%``` : Nilai kuartil kedua (Q2) atau median atau nilai tengah dari sebuah kolom
    - ```75%``` : Nilai kuartil ketiha (Q3) dari sebuah kolom
    - ```max``` : Nilai tertinggi pada sebuah kolom
  Walaupun kolom selain `review` ada yang tetap bisa diproses menggunakan fungsi `describe()` karena bertipe data `int64` dan `float64`, tetapi yang benar-benar kolom numerik hanyalah kolom `review`.

- ```python
  print(review_df['movieId'].nunique())
  ```
  Kode diatas memiliki iutput:
  ```python
  9724
  ```
  Berdasarkan output diatas, `review_df` memiliki 9724 `movieId` secara unique dari keseluruhan dataset.

- ```python
  print(review_df['userId'].nunique())
  ```
  Kode diatas memiliki iutput:
  ```python
  610
  ```
  Berdasarkan output diatas, `review_df` memiliki 610 `userId` secara unique dari keseluruhan dataset. Hal ini berarti ada 610 user yang memberikan review terhadap film-film yang mereka telah tonton.

**Visualisasi Data**

Visualisasi Data untuk `movie_df`:
- Univariate Analysis
  
  Univariate Analysis adalah jenis analisis data yang memeriksa satu variabel (atau bidang data) pada satu waktu. Tujuannya adalah untuk menggambarkan data dan menemukan pola yang ada dalam distribusi variabel tersebut. Ini termasuk penggunaan statistik deskriptif, histogram, dan box plots untuk menganalisis distribusi dan memahami sifat dari variabel tersebut.

  - Count Plot dari setiap Genre
    
    ![Count Plot - Genre](https://github.com/ensiklopedical/system-recommendation/assets/115972304/0508b4f2-4368-423a-801d-01001c134690)
    
    Gambar 1a - Count Plot Genre


  - Pie Chart dari Genre
    
    ![Pie Chart - Genre](https://github.com/ensiklopedical/system-recommendation/assets/115972304/ca14e639-b30b-48da-9906-b2622d949123)
    
    Gambar 1b - Pie Chart Genre

  Berdasarkan kedua visualisasi data diatas, terlihat bahwa genre `Comedy`, `Drama`, dan `Action` memiliki proporsi dan jumlah terbesar secara keseluruhan dibandingkan genre lainnya pada `movie_df
  
Visualisasi Data untuk `review_df`:
- Univariate Analysis
  
  Univariate Analysis adalah jenis analisis data yang memeriksa satu variabel (atau bidang data) pada satu waktu. Tujuannya adalah untuk menggambarkan data dan menemukan pola yang ada dalam distribusi variabel tersebut. Ini termasuk penggunaan statistik deskriptif, histogram, dan box plots untuk menganalisis distribusi dan memahami sifat dari variabel tersebut.

  - Count Plot dari Setiap Nilai Review
    
    ![Count Plot - Review](https://github.com/ensiklopedical/system-recommendation/assets/115972304/79298cba-dc6c-47ac-8391-7a19da4b168e)

    Gambar 1c - Count Plot Review

  - Pie Chart dari Review

    ![Pie Chart - Review](https://github.com/ensiklopedical/system-recommendation/assets/115972304/99d12b2d-fd45-418d-8c4d-adf1187772de)

    Gambar 1d - Pie Chart Genre

  Berdasarkan visualisasi data diatas, terlihat bahwa review dengan skor `4.0` dan `3.0` memiliki proporsi dan jumlah terbesar secara keseluruhan dibandingkan skor lainnya pada `review_df`.
    
  - Top 10 Movie dengan Review Terbanyak
    
    ![10 Movie Most Review](https://github.com/ensiklopedical/system-recommendation/assets/115972304/7bd680ac-84ae-4999-be99-6289faed4478)

    Gambar 1e - Top 10 Most Review Movie  

    Berdasarkan visualisasi data diatas, berikut adalah daftar `movieId` dengan review terbanyak pada dataset `review_df`.

    ```python
    top_10_movies_array = top_10_movies.index.to_numpy()
    print(top_10_movies_array)
    ```
    Berikut adalah hasilnya:
    ```python
    [ 356  318  296  593 2571  260  480  110  589  527 ]
    ```
    10 film dengan `movieId` diatas memiliki review terbanyak pada dataset.
 
## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

======================================================================================================================

Berikut ini adalah Data Preparation untuk `movie_df`:

- **Detection and Removal Duplicates**

  Data duplikat adalah baris data yang sama persis untuk setiap variabel yang ada. Dataset yang digunakan perlu diperiksa juga apakah dataset memiliki data yang sama atau data duplikat. Jika ada, maka data tersebut harus ditangani dengan menghapus data duplikat tersebut.

  **Alasan**: Data duplikat perlu didektesi dan dihapus karena jika dibiarkan pada dataset dapat membuat model Anda memiliki bias, sehingga menyebabkan overfitting. Dengan kata lain, model memiliki performa akurasi yang baik pada data pelatihan, tetapi buruk pada data baru. Menghapus data duplikat dapat membantu memastikan bahwa model Anda dapat menemukan pola yang ada lebih baik lagi.

  Berikut ini proses pendeteksian dan penghapusan nilai duplikat:
  ```python
  duplicates_movie = movie_df.duplicated()
  duplicate_count = duplicates_movie.sum()
  print(f"Number of duplicate rows: {duplicate_count}")
  ```
  Berikut adalah output-nya:

  ```python
  Number of duplicate rows: 0
  ```

  Berdasarkan hasil tersebut, tidak ditemukan adanya data duplikat, maka tidak ada juga proses penghapusannya.
  
- **Handle Missing Value**
  
  _Missing Value_ terjadi ketika variabel atau barus tertentu kekurangan titik data, sehingga menghasilkan informasi yang tidak lengkap. Nilai yang hilang dapat ditangani dengan berbagai cara seperti imputasi (mengisi nilai yang hilang dengan mean, median, modus, dll), atau penghapusan (menghilangkan baris atau kolom yang nilai hilang)

  **Alasan**: _Missing Value_ perlu ditangani karena jika dibiarkan dapat berpengaruh ke rendahnya akurasi model yang akan dibuat. Maka dari itu, penting untuk mengatasi missing value secara efisien untuk mendapatkan model _Machine Learning_ yang baik juga.

  Berikut ini adalah kode untuk mencari tahu kolom mana saja dan berapa jumlah missing value-nya:
  ```python
  movie_df.isnull().sum()
  ```
  Berikut adalah output-nya:

  ```python
  movieId    0
  title      0
  genre      0
  dtype: int64
  ```

  Berdasarkan output diatas, tidak adanya missing value pada `movie_df`
  
  
- **Delete Some Data Point**

  Pada sebuah dataset, ada saatnya beberapa baris data atau kolom perlu dihapus karena satu dan lain hal. Salah satunya agar tidak menghambat proses training dan performa dari sebuah model yang akan dibangun. Ada value pada dataset `movie_df`, khususnya kolom `genre`, yang perlu dihapus karena nama nilainya itu sendiri, yaitu `(no genres listed)'.

  **Alasan**: Hal ini perlu dilakukan karena nilai tersebut tidak mewakili genre apapun untuk sebuah film. Jika dibiarkan, ini dapat mempengaruhi performa model yang akan dibuat. Maka dari itu, baris data yang memiliki nilai ini, perlu dihapus

  Berikut adalah kodenya untuk menghapus beberapa baris data yang perlu dihapus:
  ```python
  movie_df.drop(movie_df[movie_df['genre'] == '(no genres listed)'].index, inplace=True)
  ```
  ```python
  unique_genres = movie_df['genre'].unique()
  for genre in unique_genres:
      print(genre)
  ```
  Berikut adalah output-nya:

  ```python
  Adventure
  Comedy
  Action
  Drama
  Crime
  Children
  Mystery
  Animation
  Documentary
  Thriller
  Horror
  Fantasy
  Western
  Film-Noir
  Romance
  Sci-Fi
  Musical
  War
  ```

  Berdasarkan output diatas, `(no genres listed)` terbukti sudah tidak ada lagi pada dataset

  Adapun tahap selanjutnya, yaitu penghapusan baris data yang hanya ada kurang dari 6 data point:

  Berikut ini adalah kodenya:
  ```python
  value_counts = movie_df['genre'].value_counts()
  less_than_six = value_counts[value_counts < 6]
  print("Values with less than 6 data points:")
  print(less_than_six)
  ```
  Berikut adalah output-nya:

  ```python
  Values with less than 6 data points:
  War    4
  Name: genre, dtype: int64
  ```

  Berdasarkan output diatas, `War` pada kolom `genre` hanya memiliki 4 data point. Dalam kasus ini, value yang kurang dari 6 data dalam dataset perlu dihilangkana karena tidak dapat digunakan. Hal ini akan ditindak lebih lanjut pada bagian selanjutnya.

  Berikut ini adalah kodenya:
  ```python
  movie_df = movie_df[~movie_df['genre'].str.contains('War')]
  ```
  Berhasil dilakukannya penghapusan data point pada kolon `genre` yang bernilai `War`
  

- **Changing Certain Value**
  
  Pada sebuah dataset, ada kalanya beberapa nilai perlu diproses terlebih dahulu agar proses training atau pembuatan model dapat berjalan seperti seharusnya. Salah satunya adalah pengubahan beberapa nilai yang dirasa akan mengganggu jika dibiarkan. Ada value pada dataset `movie_df`, khususnya kolom `genre`, yang perlu diganti namanya, yaitu `Sci-Fi` dan `Film-Noir`.

  **Alasan**: Hal ini perlu dilakukan karena jika dibiarkan ketika proses embedding akan terdeteksi sebagai 2 bagian yang berbeda. Maka dari itu, string dari kedua nilai tersebut harus dimodifikasi agar pada saat proses encoding tidak terpecah menjadi 2 bagian berbeda.
  
  Berikut ini adalah proses pengubahan beberapa nilai tertentu:
  ```python
  movie_df['genre'] = movie_df['genre'].replace({'Sci-Fi': 'SciFi', 'Film-Noir': 'FilmNoir'})
  ```

  Nilai `Sci-Fi` dan `Film-Noir` sudah berhasil diubah dengan menghilangkan tanda `-` pada kedua nilai tersebut. Maka dari itu, nilai tersebut sudah siap untuk diproses pada tahap selanjutnya.
  
  Berikut adalah kode untuk mengecek proses perubahan:

  ```python
  unique_genres = movie_df['genre'].unique()
  for genre in unique_genres:
      print(genre)
  ```

  Berikut ini adalah output-nya:

  ```python
  Adventure
  Comedy
  Action
  Drama
  Crime
  Children
  Mystery
  Animation
  Documentary
  Thriller
  Horror
  Fantasy
  Western
  FilmNoir
  Romance
  SciFi
  Musical
  War
  ```

  Berdasarkan output diatas, dapat dilihat bahwa sudah terlihat nilai yang baru saja diubah, yaitu `SciFi` dan `FilmNoir`

Setelah beberapa proses yang sudah dilakukan, maka `movie_df` masih memiliki:

```python
movie_df.shape
```

Outputnya:

```python
(9708, 3)
```

Setelah beberapa proses yang sudah dilakukan, maka `movie_df` masih memiliki:
- 9708 baris data
- 3 kolom data

<br>

Berikut ini adalah Data Preparation untuk `review_df` :

<br>

- **Detection and Removal Duplicates**
  
  Data duplikat adalah baris data yang sama persis untuk setiap variabel yang ada. Dataset yang digunakan perlu diperiksa juga apakah dataset memiliki data yang sama atau data duplikat. Jika ada, maka data tersebut harus ditangani dengan menghapus data duplikat tersebut.

  **Alasan**: Data duplikat perlu didektesi dan dihapus karena jika dibiarkan pada dataset dapat membuat model Anda memiliki bias, sehingga menyebabkan _overfitting_. Dengan kata lain, model memiliki performa akurasi yang baik pada data pelatihan, tetapi buruk pada data baru. Menghapus data duplikat dapat membantu memastikan bahwa model Anda dapat menemukan pola yang ada lebih baik lagi.

  Berikut ini adalah proses pendeteksian dan penghapusan data duplikatnya:

  ```python
  duplicates_review = review_df.duplicated()
  duplicate_review = duplicates_movie.sum()
  print(f"Number of duplicate rows: {duplicate_review}")
  ```

  Output-nya:
  
  ```python
  Number of duplicate rows: 0
  ```

  Berdasarkan hasil tersebut, tidak ditemukan adanya data duplikat, maka tidak ada juga proses penghapusannya.
  
  
- **Handle Missing Value**

  _Missing Value_ terjadi ketika variabel atau barus tertentu kekurangan titik data, sehingga menghasilkan informasi yang tidak lengkap. Nilai yang hilang dapat ditangani dengan berbagai cara seperti imputasi (mengisi nilai yang hilang dengan mean, median, modus, dll), atau penghapusan (menghilangkan baris atau kolom yang nilai hilang)

  **Alasan**: _Missing Value_ perlu ditangani karena jika dibiarkan dapat berpengaruh ke rendahnya akurasi model yang akan dibuat. Maka dari itu, penting untuk mengatasi missing value secara efisien untuk mendapatkan model _Machine Learning_ yang baik juga.

  Berikut ini adalah kode untuk mencari tahu kolom mana saja dan berapa jumlah missing value-nya:

  ```python
  review_df.isnull().sum()
  ```
  
  Output-nya :

  ```python
  userId       0
  movieId      0
  review       0
  timestamp    0
  ```

  Berdasarkan output diatas, tidak adanya missing value pada `review_df`. Maka, tidak perlu dilakukan pengisian pada data hilang.
  

- **Outliers Detection and Removal**

  _Outliers_ adalah titik data yang secara signifikan berbeda dari sebagian besar data dalam kumpulan data. Outliers dapat muncul karena variasi dalam pengukuran atau mungkin menunjukkan kesalahan eksperimental; dalam beberapa kasus, outliers bisa juga menunjukkan variabilitas yang sebenarnya dalam data. Penting untuk menganalisis outliers karena mereka dapat memiliki pengaruh besar pada hasil analisis statistik.
  
  Outliers adalah titik data yang secara signifikan berbeda dari sebagian besar data dalam kumpulan data. Outliers dapat muncul karena variasi dalam pengukuran atau mungkin menunjukkan kesalahan eksperimental; dalam beberapa kasus, outliers bisa juga menunjukkan variabilitas yang sebenarnya dalam data. Penting untuk menganalisis outliers karena mereka dapat memiliki pengaruh besar pada hasil analisis statistik.
  
  Proses pembersihan outliers menggunakan metode IQR (Interquartile Range) melibatkan beberapa langkah:
  
    - Menghitung Kuartil: Tentukan kuartil pertama (Q1) dan kuartil ketiga (Q3) dari data. Kuartil ini membagi data menjadi empat bagian yang sama.
    
    - Menghitung IQR: Hitung IQR dengan mengurangi Q1 dari Q3:
      $$IQR=Q3−Q1$$
    
    - Menentukan Batas Outliers:
    
      - Batas bawah untuk outliers:
        $$Q1−1.5×IQR$$
    
      - Batas atas untuk outliers:
        $$Q3+1.5×IQR$$
    
    - Identifikasi Outliers: Data yang berada di luar batas bawah dan atas ini dianggap sebagai outliers.
  
  Pembersihan Outliers yang teridentifikasi kemudian dapat dibersihkan dari dataset, baik dengan menghapusnya atau melakukan transformasi tertentu.
      
  **Alasan**:_Outliers_ perlu dideteksi dan dihapus karena jika dibiarkan dapat merusak hasil analisis statistik pada kumpulan data sehingga menghasilkan performa model yang kurang baik. Selain itu, Mendeteksi dan menghapus _outlier_ dapat membantu meningkatkan performa model _Machine Learning_ menjadi lebih baik.

  Berikut ini adalah salah satu cara mendeteksi adanya outliers atau tidak:

  ```python
  review_df['review'].describe()
  ```

  Output-nya:
  
  ```python
  count    100836.000000
  mean          3.501557
  std           1.042529
  min           0.500000
  25%           3.000000
  50%           3.500000
  75%           4.000000
  max           5.000000
  ```

  Sebelum memulai dengan proses interquartile. Perlu dilihat terlebih dahulu secara sekilas secara statistika deskriptif.

  Hanya kolom `review` yang dicek karena hanya kolom tersebut yang tergolong sebagai kolom numeric dan perlu dilakukan pemeriksaan outliers-nya.
  
  Berdasarkan output diatas, terlihbat bahwa nilai terkecil dari `review` adalah `0.5` dan terbesarnya adalah `5.0`. Kedua nilai tersebut masih di ambang wajar untuk sebuah review film. Jadi, tidak ada outliers dan tidak ada penghapus outliers untuk kolom `review`
  
- **Dropping Uneeded Column**

  Pada bagian ini adalah proses penghapusan kolom yang tidak digunakan untuk proses pembuatan model. Langkah ini diambil berdasarkan asumsi bahwa kolom yang akan dihapus tidak memberikan kontribusi terhadap prediksi yang dibuat oleh model.

  **Alasan**: Tahapan ini perlu dilakukan karena kolom yang tidak digunakan cenderung tidak memberikan informasi yang berguna untuk prediksi dan dapat menambah informasi yang tidak perlu ke dalam model. Dengan menghilangkan fitur-fitur ini, kita dapat mengurangi kompleksitas model dan mempercepat waktu pelatihan.

  Berikut ini adalah proses penghapusan kolom yang tidak diperlukan atau digunakan:

  ```python
  review_df.drop('timestamp', axis=1, inplace=True)
  ```

  Kolom `timestamp` telah berhasil dihapus. Kolom tersebut dihapus karena tidak diperlukan untuk proses pembuatan sistem rekomendasi secara collaborative filtering.

  Berikut dilakukan pengecekan ukuran dari `review_df`:

  ```python
  review_df.shape
  ```

  Output-nya:
  
  ```python
  (100836, 3)
  ```

  Berdasarkan output tersebut, maka `review_df` masih memiliki:
    - 100836 baris data
    - 3 kolom data
  
- **Encoding**

  Encoding adalah proses konversi informasi dari satu bentuk atau format ke bentuk lain, yang sering kali dilakukan untuk memastikan kompatibilitas dan pemrosesan yang tepat oleh berbagai sistem komputer. Proses ini sangat penting dalam dunia digital, di mana berbagai jenis data, seperti teks, gambar, dan suara, harus diubah menjadi format yang dapat dipahami oleh perangkat keras dan perangkat lunak.


  **Alasan:** Tahap ini perlu dilakukan karena Encoding memungkinkan data dari berbagai sumber dan format untuk diubah menjadi format standar yang dapat dipahami dan memastikan bahwa informasi dapat diproses


- **Train Test Split**

  Train Test Split adalah metode yang digunakan untuk membagi dataset menjadi dua bagian: satu untuk melatih model (_training set_) dan satu lagi untuk menguji model (_testing set_). Biasanya, data dibagi dengan proporsi tertentu, misalnya 80% untuk training dan 20% untuk testing.

  **Alasan**: Proses ini dilakukan agar dapat mengevaluasi kinerja model secara objektif. Dengan memisahkan data uji, kita dapat mengukur seberapa baik model memprediksi data baru yang tidak pernah dilihat sebelumnya, yang merupakan indikator penting dari kemampuan generalisasi model.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
