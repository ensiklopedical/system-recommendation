# Laporan Proyek Machine Learning - Faisal Ahmad Gifari

## Project Overview

Platform streaming film telah menjadi bagian integral dari hiburan saat ini. Dengan akses internet yang semakin luas, platform-platform tersebut memberikan kemudahan bagi penggunanya dalam menikmati berbagai film dari seluruh dunia. Adanya platform streaming memungkinkan penonton untuk memilih konten yang ingin ditonton kapan saja dan di mana saja[[1](https://www.emerald.com/insight/content/doi/10.1108/INTR-11-2021-0861/full/html)]. Hal ini memberikan fleksibilitas yang tidak dapat ditawarkan oleh bioskop atau televisi tradisional.  Namun, untuk meningkatkan pengalaman pengguna dalam menggunakan layanan tersebut juga perlu diperhatikan. Jika dibiarkan, penonton bisa saja tidak ingin lagi menggunakan layanan platform streaming tersebut atau memberhentikan langganannya. Maka dari itu, salah satu solusi dari masalah ini adalah pembuatan sistem rekomendasi.

Salah satu contoh nyata efektivitas sistem rekomendasi adalah keberhasilan Netflix dalam mempertahankan dan meningkatkan basis pelanggannya. Dengan menggunakan sistem rekomendasi yang canggih, Netflix dapat menganalisis preferensi pemirsa dan perilaku menonton untuk memberikan saran film dan acara TV yang relevan[[2](https://www.ssdjournal.org/DergiDetay.aspx?ID=931&Detay=Ozet)]. Hasilnya, sekitar 75% tontonan di Netflix berasal dari rekomendasi sistem. Hal ini menunjukkan betapa pentingnya sistem rekomendasi dalam membantu pengguna menemukan konten yang mereka sukai, yang pada akhirnya meningkatkan kepuasan dan loyalitas pengguna terhadap platform

Sistem rekomendasi telah terbukti menjadi salah satu bagian utama yang meningkatkan pengalaman pengguna di platform streaming[[3](https://dl.acm.org/doi/10.1145/3565472.3592960)]. Sistem menggunakan algoritma untuk menganalisis preferensi pengguna dan memberikan saran film yang relevan[[4](https://www.nature.com/articles/s41598-023-34192-x)]. Sistem rekomendasi memberikan dampak yang signifikan bagi pengguna atau penonton karena membantu pengguna menemukan film yang sesuai dengan seleranya tanpa harus mencari terlebih dahulu, sehingga meningkatkan kepuasan pengguna dan potensi waktu menonton.

Ada beberapa pendekatan yang dapat digunakan dalam membuat model sistem rekomendasi film. Pendekatan berbasis konten memanfaatkan fitur-fitur seperti genre untuk merekomendasikan film serupa [[5](https://www.irjmets.com/uploadedfiles/paper//issue_6_june_2023/42626/final/fin_irjmets1687806612.pdf)]. Sementara itu, pendekatan pemfilteran kolaboratif mengumpulkan dan menganalisis data dari banyak pengguna untuk menemukan pola dan preferensi bersama[[6](https://ieeexplore.ieee.org/document/10142424)]. Perkembangan model ini terus berkembang seiring dengan kemajuan teknologi dan analisis data.

## Business Understanding

Pengguna dari platform streaming film dapat berhenti menggunakan sebuah layanan streaming film jika pengalaman yang didapatkannya tidak begitu baik. Salah satu untuk meningkatkan pengalaman pengguna paltform streaming film adalah dengan adanya sistem rekomendasi agar penonton bisa mendapatkan konten yang sesuai dengan preferensinya atau riwayat tontonannya. Ada beberapa pendekatan yang dapat digunakan untuk membangun model sistem rekomendasi. Salah satunya adalah sistem rekomendasi dengan content-based learning yang memberikan rekomendasi berdasarkan item serupa. Kemudian, sistem rekomendasi dengan collaborative filtering yang bekerja dengan cara mengumpulkan dan menganalisis data dari banyak pengguna untuk menemukan pola dan preferensi secara kolektif. Dengan adanya sistem rekomendasi pada platfom streaming film, pengguna dapat mendapatkan pengalaman pengguna yang baik.

### Problem Statements

- Bagaimana memahami dan mengetahui terkait data dari datasey digunakan untuk pembuatan model sistem rekomendasi?
- Bagaimana membuat model sistem rekomendasi dengan pendekatan content-based filtering?
- Bagaimana membuat model sistem rekomendasi dengan pendekatan collaborative filtering?
- Bagaimana cara mengukur performa model sistem rekomendasi yang sudah dibuat?

### Goals

- Melakukan langkah-langkah untuk memahami dataset terlebih dahulu, seperti EDA dan Visualisasi Data.
- Membuat sistem rekomendasi film dengan content-based filtering.
- Membuat sistem rekomendasi film dengan collaborative filtering.
- Melakukan evaluasi terhadap model sistem rekomendasi yang telat dibuat.

### Solution Approach

- Melakukan EDA untuk mengeksplorasi fitur menggunakan fungsi `shape`, `key`, `info` pada dataset. Kemudian, dilakukan visualisasi data seperti count plot dan pie chart untuk mendapatkan gambaran atau ilustrasu lebih jelas mengenai dataset yang digunakan.
- Membangun sistem rekomendasi dengan content-based filtering yang memberikan rekomendasi kepada pengguna berdasarkan kesamaan pada item yang ada. Data yang digunakan berisi data dari genre dari setiap film. Dataset tersebut juga melewati tahap Data Preparation agar dataset dapat digunakan untuk proses pembangunan model seperti, menangani data duplikat, missing value, dan mengganti beberapa data agar sesuai. Kemudian, data yang sudah siap, diproses ke tahap modelling yang memanfaatkan `Tfidvectorizer`, `cosine similarity`, dan fungsi buatan yang mengembalikan rekomendasi berdasarkan kesamaan pada item. Pendekatan ini berfokus pada karakteristik atau konten dari item yang direkomendasikan
- Membangun sistem rekomendasi dengan colaborative filtering yang memberikan rekomendasi kepada pengguna dengan menganalisis perilaku dan preferensi pengguna. Data yang digunakan berisi data review untuk film-film dari user. Dataset tersebut juga melewati tahap Data Preparation agar dataset dapat digunakan untuk proses pembangunan model seperti, menangani data duplikat, missing value, encoding, dan train test split. Kemudian, data yang sudah siap, diproses ke tahap modelling yang menggunakan `RecommenderNet` dan `Early Stopper` dalam proses training-nya. Pendekatan ini membutuhkan data terkait user. 
- Melakukan perhitungan skor presisi untuk mengukur performa dari model sistem rekomendasi film dengan content-based learning. Kemudian, menggunakan skor RMSE atau root mean squared error untuk mengukur performa dari model sistem rekomendasi film dengan colaborative filtering.



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

Data Preparation adalah proses pembersihan, transformasi, dan pengorganisasian data mentah ke dalam format yang dapat dipahami oleh algoritma machine learning. Bagian ini menjelaskan urutan langkah-langkah Data Preparation yang dilakukan beserta penjelasan dan alasannya untuk setiap dataset.

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

  Berikut ini adalah proses dari encoding yang dilakukan:

  - Encoding `userId`
    ```python
    user_id = review_df['userId'].unique().tolist() # Mengubah userId menjadi list tanpa nilai yang sama
    user_to_user = {x: i for i, x in enumerate(user_id)} # Melakukan encoding userId
    user_encode_to_user = {i: x for i, x in enumerate(user_id)} # Melakukan proses encoding angka ke ke userId
    
    print('list userId :  ', user_id)
    print('encoded userId :  ', user_to_user)
    print('encoded angka ke userId :  ', user_encode_to_user)
    ```
    Encoding `userId` berhasil dilakukan. Output tersedia di notebook dan tidak dapat ditampilkan disini karena terlalu panjang.
  
  - Encoding `movieId`
    ```python
    movie_id = review_df['movieId'].unique().tolist() # Mengubah movieId menjadi list tanpa nilai yang sama
    movie_to_movie = {x: i for i, x in enumerate(movie_id)} # Melakukan proses encoding movieId
    movie_encode_to_movie = {i: x for i, x in enumerate(movie_id)} # Melakukan proses encoding angka ke movieId
    
    print('list movieId :  ', movie_id)
    print('encoded movieId :  ', movie_to_movie)
    print('encoded angka ke movieId :  ', movie_encode_to_movie)
    ```
    Encoding `movieId` berhasil dilakukan. Output tersedia di notebook dan tidak dapat ditampilkan disini karena terlalu panjang.

  - Mapping hasil encoding `userId` dan `movieId`:
    ```python
    review_df['user'] = review_df['userId'].map(user_to_user) # Mapping userId ke dataframe user
    review_df['movie'] = review_df['movieId'].map(movie_to_movie) # Mapping movieId ke dataframe resto
    ```
  
    Hasil encoding tadi, di-mapping ke dalam dataframe `review_df` dengan menempati kolom baru untuk masing-masing hasil.
  
    Berikut dilakukan pengecekan pada `review_df`:
  
    ```python
    review_df.head(5)
    ```
  
    Proses mapping berhasil dilakukan karena sudah terdapat dua kolom baru, yaitu `user` dan `movie`

  - Berikut adalah pengecekan kembali pada `movie_df` dari beberapa aspek lainnya:

    ```python
    num_users = len(user_to_user) # Mendapatkan jumlah user
    num_movie = len(movie_to_movie) # Mendapatkan jumlah review
    min_review = min(review_df['review']) # Nilai minimum review
    max_review = max(review_df['review']) # Nilai maksimal review
    
    print('total user: {}'.format(num_users))
    print('total review: {}'.format(num_movie))
    print('MIN review: {}'.format(min_review))
    print('MAX review: {}'.format(max_review))
    ```
  
    Outputnya:
  
    ```python
    total user: 610
    total review: 9724
    MIN review: 0.5
    MAX review: 5.0
    ```
  
    Berdasarkan output diatas, dapat dilihat bahwa pada `review_df` terdapat:
      - total user: 610
      - total review: 9724
      - MIN review: 0.5
      - MAX review: 5.0

 
  Proses encoding telah berhasil dilakukan


- **Train Test Split**

  Train Test Split adalah metode yang digunakan untuk membagi dataset menjadi dua bagian: satu untuk melatih model (_training set_) dan satu lagi untuk menguji model (_testing set_). Biasanya, data dibagi dengan proporsi tertentu, misalnya 80% untuk training dan 20% untuk testing.

  **Alasan**: Proses ini dilakukan agar dapat mengevaluasi kinerja model secara objektif. Dengan memisahkan data uji, kita dapat mengukur seberapa baik model memprediksi data baru yang tidak pernah dilihat sebelumnya, yang merupakan indikator penting dari kemampuan generalisasi model.

  Berikut ini adalah proses Train Test  Split yang dilakukan:

  - Dilakukan pengacakan pada dataset agar teracak merata
    
    ```python
    review_df = review_df.sample(frac=1, random_state=18)
    ```
    
  - Pemisahan bagian atribur dan label ke dua variabel

    ```python
    x_df = review_df[['user', 'movie']].values # Membuat variabel x_df untuk mencocokkan data user dan movie menjadi satu value
    y_df = review_df['review'].apply(lambda x: (x - min_review) / (max_review - min_review)).values # Membuat variabel y_df untuk membuat review dari hasil
    ```

    Pemisahan `review_df` menjadi dua bagian ke `x_df` dan `y_df` untuk proses Train Test Split berhasil dilakukan
    
  - Split
 
    ```python
    train_indices = int(0.9 * review_df.shape[0])
    x_train, x_val, y_train, y_val = (
        x_df[:train_indices],
        x_df[train_indices:],
        y_df[:train_indices],
        y_df[train_indices:]
    )
    ```
    Proses Train Test Split telah dilakukan ke empat variabel berbebeda dengan komposisi 0.9 untuk train dan 0.1 untuk val. Berikut adalah keempatnya:
    - x_train
    - x_val
    - y_train
    - y_val
    
 
  Proses Train Test Split berhasil dilakukan. 
    

## Modeling

Model yang dibuat terdiri dari dua model dengan algoritma dan pendekatan yang berbeda, yaitu `Content-Based Filtering` dan `Collaborative Filtering`. `Content-Based Filtering` menggunakan dataset `movie_df` dan `Collaborative Filtering` menggunakan dataset `review_df`. Kedua model atau algoritma tersebut memiliki pendekatan yang berbeda-beda. Berikut ini adalah penjelasan berserta kelebihan dan kekurangan dari keduanya:

- **Content-Based Filtering**
  
  Content-Based Filtering adalah metode yang digunakan dalam sistem rekomendasi untuk memberikan saran kepada pengguna berdasarkan item-item yang telah mereka sukai atau pilih sebelumnya. Metode ini berfokus pada karakteristik atau konten dari item yang ingin direkomendasikan.

  **Kelebihan Content-Based Filtering:**
  
  - **Personalisasi**: Dapat memberikan rekomendasi yang sangat personal karena didasarkan pada preferensi sebelumnya dari pengguna itu sendiri.
  - **Transparansi**: Mudah untuk menjelaskan mengapa suatu item direkomendasikan, karena rekomendasi didasarkan pada fitur-fitur item yang telah disukai pengguna.

  **Kekurangan Content-Based Filtering:**
  
  - **Keterbatasan Diversifikasi**: Cenderung merekomendasikan item yang mirip dengan yang sudah diketahui pengguna, sehingga kurang memberikan kejutan atau item baru yang berbeda.
  - **Ketergantungan pada Konten:** Memerlukan data yang cukup tentang konten item untuk bekerja dengan baik, dan kualitas rekomendasi sangat bergantung pada kualitas deskripsi item tersebut.

  Pendekatan ini menggunakan atribut-atribut atau fitur-fitur item untuk menentukan kesamaan antara item yang ada. Dalam konteks proyek ini, content-based filtering akan memberikan rekomendasi film berdasarkan `genre` dari film yang ada dari dataset `movie_df`. Model akan memberikan rekomendasi film-film yang memiliki genre yang sama berdasarkan genre dari judul film yang digunakan sebagai input.

- **Collaborative Filtering**
  
  Collaborative Filtering adalah teknik yang digunakan dalam sistem rekomendasi untuk memberikan saran kepada pengguna berdasarkan preferensi atau perilaku pengguna lain yang memiliki kesamaan. Teknik ini mengumpulkan dan menganalisis sejumlah besar informasi tentang perilaku pengguna, aktivitas, atau preferensi dan memprediksi apa yang pengguna akan suka berdasarkan kesamaan dengan pengguna lain.

  **Kelebihan Collaborative Filtering:**
  
  - **Diversifikasi Rekomendasi**: Dapat memberikan rekomendasi yang beragam karena didasarkan pada preferensi dari banyak pengguna.
  - **Tidak Bergantung pada Konten**: Tidak memerlukan pengetahuan tentang konten item, sehingga dapat bekerja dengan item yang memiliki sedikit atau tanpa data konten sama sekali.


  **Kekurangan Collaborative Filtering:**

  - **Masalah Cold Start**: Sulit untuk memberikan rekomendasi kepada pengguna baru atau untuk item baru yang belum memiliki data interaksi.
  - **Scalability**: Dapat menjadi tantangan ketika jumlah pengguna dan item sangat besar karena membutuhkan komputasi yang intensif.
  - **Collaborative** Filtering bekerja dengan baik ketika ada cukup data dari pengguna, tetapi bisa menjadi kurang efektif jika data tersebut jarang atau tidak lengkap. Oleh karena itu, sering kali digunakan dalam kombinasi dengan teknik lain untuk meningkatkan kinerja sistem rekomendasi.

  Pendekatan ini menggunakan atribut-atribut atau fitur-fitur yang ada pada dataset `review_df` untuk memberikan rekomendasi kepada seorang user. Sistem Rekomendasi yang dibuat memberikan rekomendasi berdasarkan skor `review` yang diberikan dari sebuah film dan `genre` yang dilakukan oleh seorang user. Lebih tepatnya, 5 film dengan skow `review` tertinggi dan setiap film tersebut memiliki genrenya masing-masing. Kemudian, model akan memberikan 10 rekomendasi film untuk user tersebut berdasarkan riwayat review user tersebut.

Berikut ini adalah proses _Modelling and Result_ dari kedua algoritma tersebut:

- _Modelling and Result_ **Content-Based Filtering**

  - Modelling

    - Inisiasi `TfidVectorizer`
   
      ```python
      tf_id = TfidfVectorizer()
      tf_id.fit(movie_df['genre'])
      tf_id.get_feature_names_out()
      ```
      
      Output-nya:

      ```python
      array(['action', 'adventure', 'animation', 'children', 'comedy', 'crime',
       'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical',
       'mystery', 'romance', 'scifi', 'thriller', 'war', 'western'],
      dtype=object)
      ```

      Output diatas adalah array yang berisi nilai-nilai yang ada pada kolom `genre`
      
  
    - `fit_tranform` dan pengecekan ukuran
   
      ```python
      tfidf_matrix = tf_id.fit_transform(movie_df['genre'])
      tfidf_matrix.shape 
      ```

      Outputnya:
      
      ```python
      (9708, 18)
      ```

      Berdasarkan output diatas, dapat dilihat bahwa ukuran matriksnya sebesar 9708 x 18
      
    - `to_dense()`
   
      ```python
      tfidf_matrix.todense()
      ```
   
      Outputnya:
      
      ```python
      matrix([[0., 1., 0., ..., 0., 0., 0.],
              [0., 1., 0., ..., 0., 0., 0.],
              [0., 0., 0., ..., 0., 0., 0.],
              ...,
              [0., 0., 0., ..., 0., 0., 0.],
              [1., 0., 0., ..., 0., 0., 0.],
              [0., 0., 0., ..., 0., 0., 0.]])
      ```
      
      Berdasarkan output diatas, proses operasi menggunakan `todense()` sudah berhasil dilakukan
      
    - Pembuatan dataframe dari matrix tf-idf
   
      ```python
      # Membuat dataframe untuk melihat tf-idf matrix
      pd.DataFrame(
          tfidf_matrix.todense(),
          columns=tf_id.get_feature_names_out(),
          index=movie_df.title
      ).sample(18, axis=1).sample(7, axis=0)

      ```
      Dataframe berhasil dibuat dengan data dari matriks yang sudah dibuat sebelumnya
  
    - `cosine_similarity()`
   
      ```python
      # Proses perhitungan cosine_similarity
      cosine_sim = cosine_similarity(tfidf_matrix)
      cosine_sim
      ```
   
      Outputnya:
      
      ```python
      array([[1., 1., 0., ..., 0., 0., 0.],
             [1., 1., 0., ..., 0., 0., 0.],
             [0., 0., 1., ..., 0., 0., 1.],
             ...,
             [0., 0., 0., ..., 1., 0., 0.],
             [0., 0., 0., ..., 0., 1., 0.],
             [0., 0., 1., ..., 0., 0., 1.]])
      ```
      Berdasarkan output diatas, proses perhitungan `cosine_similarity` telah berhasil dilakukan.
      
    - Pembuatan dataframe dari `cosine_sim`
   
      ```python
      cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_df['title'], columns=movie_df['title'])
      print('Ukuran Dataframe : ', cosine_sim_df.shape)
      ```
   
      Outputnya:
      
      ```python
      Ukuran Dataframe :  (9708, 9708)
      ```
      Berdasarkan output diatas, proses pembuatan dataframe berhasil dilakukan dan dataframe memiliki ukuran 9708 x 9708.
      
    - Similarity matrix pada data
   
      ```python
      cosine_sim_df.sample(5, axis=1).sample(7, axis=0)
      ```
      
    - Pembuatan function `movie_recommendations()`
   
      ```python
      def movie_recommendations(title, similarity_data=cosine_sim_df, items=movie_df[['title', 'genre']], k=5):
      index = similarity_data.loc[:,title].to_numpy().argpartition(range(-1, -k, -1))
      closest_data = similarity_data.columns[index[-1:-(k+2):-1]]
      closest_data = closest_data.drop(title, errors='ignore')
  
      return pd.DataFrame(closest_data).merge(items).head(k)
      ```
   
      Function utama yang digunakan untuk pembuatan model Content Based telah berhasil dibuat
  
    
  - Result
 
    Untuk contoh atau simulasi penggunaan model, kita gunakan `Train to Busan (2016)` yang ber-genre `Action`
    
    ```python
    movie_df[movie_df.title.eq('Train to Busan (2016)')]
    ```
 
    Outputnya:
 
    
    | movieId | title                  | genre  |
    |---------|------------------------|--------|
    | 9364    | Train to Busan (2016)  | Action |    
    
 
    Kemudian, memanggil `movie_recommendations` untuk mendapatkan `Top-N Recommendations`
 
    ```python
    recommendations_result = movie_recommendations('Train to Busan (2016)')
    recommendations_result
    ```

    Outputnya;
 
    
    |   | title                                      | genre  |
    |---|--------------------------------------------|--------|
    | 0 | Django Unchained (2012)                    | Action |
    | 1 | Collision Course (1989)                    | Action |
    | 2 | Family, The (2013)                         | Action |
    | 3 | Highlander: Endgame (Highlander IV) (2000) | Action |
    | 4 | Saint, The (1997)                          | Action |
    

    Berikut ini adalah hasil dari `Top-N Recommendation` menggunakan Content-Based Filterting. Proses penggunaan model berhasil dilakukan dan model dapat memberikan hasil rekomendasi berdasarkan input yang diberikan.

    Pada contoh diatas, model berhasil memberikan rekomendasi film yang juga ber-genre `Action` berdasarkan input yang diberikan, yaitu `Train to Busan (2016)` yang juga bergenre `Action`

**Model telah dapat berfungsi dengan baik**.
    
  
- _Modelling and Result_ **Collaborative Filtering**

  - Modelling 

    - Pembuatan `class` `RecommenderNet`
  
      ```python
      class RecommenderNet(Model):
    
      def __init__(self, num_users, num_movie, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movie = num_movie
        self.embedding_size = embedding_size
    
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1) 
    
        self.movie_embedding = layers.Embedding(
            num_movie,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer = keras.regularizers.l2(1e-6)
        )
    
        self.movie_bias = layers.Embedding(num_movie, 1) 
    
      def call(self, inputs):
        user_vector = self.user_embedding(inputs[:,0]) 
        user_bias = self.user_bias(inputs[:, 0]) 
        movie_vector = self.movie_embedding(inputs[:, 1]) 
        movie_bias = self.movie_bias(inputs[:, 1]) 
    
        dot_user_movie = tensorflow.tensordot(user_vector, movie_vector, 2)
    
        x = dot_user_movie + user_bias + movie_bias
    
        return tensorflow.nn.sigmoid(x) 
      ```
  
      `class` `RecommenderNet` yang digunakan untuk pembuatan model Collaborative Filtering telah berhasil dibuat.
  
    - Inisiasi Model
   
      ```python
      model = RecommenderNet(num_users, num_movie, 50) # inisialisasi model
      model.compile(
          loss = keras.losses.BinaryCrossentropy(),
          optimizer = keras.optimizers.Adam(learning_rate=0.001),
          metrics=[keras.metrics.RootMeanSquaredError()]
      )
      ```
  
      Inisiasi model terlah berhasil dilakukan
   
    - Early Stopper
   
      ```python
      early_stopper = EarlyStopping(monitor='val_root_mean_squared_error',
                                patience=5,
                                verbose=1,
                                restore_best_weights=True)
      ```
    
      Inisiasi Callback Early Stopper yang akan memantau proses training model. Model akan berhenti jika `val_root_mean_squared_error` tidak mengalami penurunan lagi selama 5 epochs. Setelah berhenti, model pada epoch tertentu yang memiliki performa terbaik akan dipertahankan
   
      
    - Training
   
      ```python
      history = model.fit(
            x = x_train,
            y = y_train,
            batch_size = 8,
            epochs = 100,
            callbacks = [early_stopper],
            validation_data = (x_val, y_val)
      )
      ```
  
      Berikut ini hasil proses training yang sudah selesai pada epochs ke- :
  
      ```python
  
      ```
      
  - Result

    ```python
    user_id = review_df.userId.sample(1).iloc[0]
    movie_reviewed_by_user = review_df[review_df.userId == user_id]
    movie_not_reviewed = review_df[~review_df['movieId'].isin(movie_reviewed_by_user.movieId.values)]['movieId']
    movie_not_reviewed = list(
        set(movie_not_reviewed)
        .intersection(set(movie_to_movie.keys()))
    )
    movie_not_reviewed = [[movie_to_movie.get(x)] for x in movie_not_reviewed]
    user_encoder = user_to_user.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movie_not_reviewed), movie_not_reviewed)
    )
    ```
    
    ```python
    reviews = model.predict(user_movie_array).flatten()

    top_reviews_indices = reviews.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encode_to_movie.get(movie_not_reviewed[x][0]) for x in top_reviews_indices
    ]
    
    print('List recommendations movie untuk users : {}'.format(user_id))
    print('====' * 9)
    print('Movie dengan skor review tinggi dari user ')
    print('=====' * 8)
    
    top_movie_user = (
        movie_reviewed_by_user.sort_values(
            by = 'review',
            ascending=False
        )
        .head(5)
        .movieId.values
    )
    
    movie_df_rows = movie_df[movie_df['movieId'].isin(top_movie_user)]
    for row in movie_df_rows.itertuples():
        print(row.title, ':', row.genre)
    
    print('====' * 8)
    print('Top 10 movie recommendation')
    print('====' * 8)
    
    recommended_movie = movie_df[movie_df['movieId'].isin(recommended_movie_ids)]
    for row in recommended_movie.itertuples():
        print(row.title, ':', row.genre)
    ```

    Berikut ini adalah output-nya:

    ```python
    List recommendations movie untuk users : 567
    ====================================
    Movie dengan skor review tinggi dari user 
    ========================================
    Eraserhead (1977) : Drama
    Come and See (Idi i smotri) (1985) : Drama
    Jetée, La (1962) : Romance
    There Will Be Blood (2007) : Drama
    It's Such a Beautiful Day (2012) : Animation
    ================================
    Top 10 movie recommendation
    ================================
    Shawshank Redemption, The (1994) : Crime
    Rear Window (1954) : Mystery
    North by Northwest (1959) : Action
    Casablanca (1942) : Drama
    Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) : Drama
    Citizen Kane (1941) : Drama
    Rebecca (1940) : Drama
    Notorious (1946) : FilmNoir
    To Catch a Thief (1955) : Crime
    Lawrence of Arabia (1962) : Adventure
    ```

    Hasil diatas adalah hasil dari `Top-N Recommendation` menggunakan Collaborative Filterting. Proses penggunaan model berhasil dilakukan dan model dapat memberikan hasil rekomendasi berdasarkan review dari user tertentu dan memberikan rekomendasi film lainnya yang cocok untuk user tersebut.

    Pada contoh diatas, model berhasil memberikan rekomendasi film untuk user nomor `567` yang pernah memberikan skor review tinggi ke film dan genre:
  - `Eraserhead (1977) : Drama`
  - `Come and See (Idi i smotri) (1985) : Drama`
  - `Jetée, La (1962) : Romance`
  - `There Will Be Blood (2007) : Drama`
  - `It's Such a Beautiful Day (2012) : Animation`
  
  Model memberikan 10 rekomendasi berupa film dengan genre:
  - `Shawshank Redemption, The (1994) : Crime`
  - `Rear Window (1954) : Mystery`
  - `North by Northwest (1959) : Action`
  - `Casablanca (1942) : Drama`
  - `Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) : Drama`
  - `Citizen Kane (1941) : Drama`
  - `Rebecca (1940) : Drama`
  - `Notorious (1946) : FilmNoir`
  - `To Catch a Thief (1955) : Crime`
  - `Lawrence of Arabia (1962) : Adventure`

  **Model telah dapat berfungsi dengan cukup baik**. 

## Evaluation
Untuk mengukur bagaimana performa dari model yang telah dibuat, diperlukannya metriks evaluasi untuk mengevaluasi model sistem rekomendasi film. Berikut adalah rincian metrik yang digunakan untuk tiap pendekatan:

- `Content-Based Filtering` : `Precision`
- `Collaborative Filtering` : `Root Mean Squared Error`

Berikut ini adalah penjelasan mengenai setiap metrik beserta hasil perhitungan metrik dari model yang telah dibuat :

[JANGAN LUPA KODENYA]

- `Content-Based Filtering` : `Precision`
  - `Precision`
  
    Presisi merupakan ukuran yang menilai efektivitas model klasifikasi dalam mengidentifikasi label positif. Ukuran ini merupakan perbandingan antara jumlah prediksi yang benar-benar positif dengan keseluruhan hasil yang diprediksi sebagai positif, termasuk yang sebenarnya negatif.

    Berikut adalah formula dan cara kerja dari `Precision` :
    
    - Formula

      $$Precision = TP/(TP+FP)$$

      Dalam Konteks sistem rekomendasi menjadi:

      ![Precision](https://github.com/ensiklopedical/system-recommendation/assets/115972304/efd048df-2997-4808-addc-da64f4d34469)
      

    - Cara Kerja

      Formula tersebut mengukur presisi dalam konteks sistem rekomendasi. Presisi dihitung dengan membagi jumlah rekomendasi yang relevan dengan jumlah total item yang direkomendasikan. Jadi, jika sebuah sistem merekomendasikan 10 film dan hanya 6 yang relevan atau disukai oleh pengguna, maka presisi sistem tersebut adalah 0.6 atau 60%. Ini menunjukkan seberapa akurat sistem dalam memberikan rekomendasi yang sesuai dengan kebutuhan atau selera pengguna.
      
  - Hasil `Precision` dari model `Content-Based Learning`

    
- `Colaborative Filtering` : `Root Mean Squared Error`
  - `Root Mean Squared Error`
    
    Root Mean Square Error (RMSE) adalah metrik yang sering digunakan dalam machine learning untuk mengukur seberapa baik sebuah model prediktif dapat memperkirakan nilai yang sebenarnya. RMSE merupakan akar kuadrat dari rata-rata perbedaan kuadrat antara nilai yang diprediksi oleh model dan nilai yang sebenarnya (nilai aktual).

    Berikut ini adalah formula dan cara kerja dari `Root Mean Squared Error` :

    - Formula
   
      $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   
    - Cara Kerja
    - 
      RMSE menghitung akar kuadrat dari rata-rata perbedaan kuadrat antara nilai yang diprediksi oleh model dan nilai sebenarnya. Proses kerjanya melibatkan beberapa langkah. Pertama, untuk setiap titik data, kita menghitung selisih antara prediksi model dan nilai aktual. Selisih ini kemudian dikuadratkan untuk menghilangkan nilai negatif dan memberikan bobot lebih pada kesalahan yang lebih besar. Setelah itu, kita menghitung rata-rata dari nilai-nilai kuadrat tersebut. Terakhir, kita mengambil akar kuadrat dari rata-rata ini untuk mendapatkan RMSE.
    
  - hasil nya


## Referensi

[1] Q. Yang, J. Huo, H. Li, Y. Xi, and Y. Liu, “Can social interaction-oriented content trigger viewers’ purchasing and gift-giving behaviors? Evidence from live-streaming commerce,” Internet Research, Mar. 2023, doi: https://doi.org/10.1108/intr-11-2021-0861.

[2] Şeyma BOZKURT UZAN and Kutluk ATALAY, “DEVELOPING NEW SUGGESTIONS FOR THE CONTENTS OF A DIGITAL PLATFORM USING RECOMMENDATION SYSTEMS ALGORITHMS,” Social science development journal, vol. 8, no. 38, pp. 187–202, Jul. 2023, doi: https://doi.org/10.31567/ssd.931.

[3] Karlijn Dinnissen and C. Bauer, “Amplifying Artists’ Voices: Item Provider Perspectives on Influence and Fairness of Music Streaming Platforms,” Jun. 2023, doi: https://doi.org/10.1145/3565472.3592960.

[4] P. Khambatta, S. Mariadassou, J. Morris, and S. C. Wheeler, “Tailoring recommendation algorithms to ideal preferences makes users better off,” Scientific Reports, vol. 13, p. 9325, Jun. 2023, doi: https://doi.org/10.1038/s41598-023-34192-x.

[5] “Movie Recommendation Systems Using Content-Based Filtering,” International Research Journal of Modernization in Engineering Technology and Science, Jun. 2023, doi: https://doi.org/10.56726/irjmets42626.
‌
[6] S. Katkam, A. Atikam, P. Mahesh, M. Chatre, S. S. Kumar, and S. G. R, “Content-based Movie Recommendation System and Sentimental analysis using ML,” IEEE Xplore, May 01, 2023. https://ieeexplore.ieee.org/document/10142424
