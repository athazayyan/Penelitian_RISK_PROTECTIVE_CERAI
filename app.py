# ==========================================================
# app_skripsi.py — Dashboard Penelitian Perceraian
# Tujuan 1: Prediksi, Tujuan 2: Faktor Dominan, Tujuan 3: Interaksi
# Jalankan: streamlit run app_skripsi.py
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Dashboard Penelitian Perceraian",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# PATH DEFAULT
# ==========================================================
DEFAULT_MODEL = "rf_smote30_bestmodel_20261002_1913.joblib"
DEFAULT_OOF   = "oof_probabilities_20261002_1913.csv"
DEFAULT_DATA  = "yang_sudah_dibersihkan.csv"

# ==========================================================
# DECODERS
# ==========================================================
DECODE_PENDIDIKAN = {1: "SD/SMP/SMA/MAN", 2: "D-III/D-IV/S1", 3: "S2/S3"}
DECODE_DPS = {0: "Tidak pernah", 1: "Jarang", 2: "Kadang-kadang",
              3: "Sering", 4: "Selalu"}
DECODE_KDRT = {1: "Sangat Tidak Setuju", 2: "Tidak Setuju",
               3: "Setuju", 4: "Sangat Setuju"}
DECODE_NIKAH = {0: "Menikah", 1: "Cerai Hidup / Pernah Bercerai"}
DECODE_PRED  = {0: "Tidak", 1: "Ya"}

# ==========================================================
# KELOMPOK ASPEK
# ==========================================================
DPS_GROUPS = {
    "Shared meaning and forgiveness (1-20)":  list(range(1, 21)),
    "Love maps (21-30)":                      list(range(21, 31)),
    "Negative Conflict Behaviors (31-41)":    list(range(31, 42)),
    "Stonewalling (42-47)":                   list(range(42, 48)),
    "Defensiveness (48-54)":                  list(range(48, 55)),
}
FS_GROUPS = {
    "Financial Distress / Well-Being (1-8)": list(range(1, 9)),
}
KDRT_GROUPS = {
    "Fisik - Favorable":          [1, 7, 13, 29],
    "Fisik - Unfavorable":        [6, 17, 20, 30, 32, 36],
    "Seksual - Favorable":        [24, 25, 39],
    "Seksual - Unfavorable":      [12, 14, 19, 31, 37, 40],
    "Memata-matai - Favorable":   [2, 3, 9],
    "Memata-matai - Unfavorable": [15, 23, 28, 33, 38],
    "Psikologis - Favorable":     [5, 8, 10, 16, 26, 34],
    "Psikologis - Unfavorable":   [4, 11, 18, 21, 22, 27, 35],
}

# ==========================================================
# KOLOM SESUAI CSV
# ==========================================================
KOLOM_DPS = [
    '1. Jika pertengkaran saya dan pasangan mulai memanas, dan salah satu dari kita meminta maaf maka masalah tidak akan berlarut.',
    '2. Ketika keadaan menjadi sulit, saya tahu bahwa saya dan pasangan dapat  mengabaikan perbedaan diantara kami.',
    '3. Dalam situasi tertentu, saya dan pasangan dapat mendiskusikan kembali masalah yang diperdebatkan dan memperbaikinya.',
    '4. Saat terjadi pertengkaran, saya percaya bahwa komunikasi dengan pasangan pada akhirnya dapat menyelesaikan masalah.',
    '5. Waktu yang saya habiskan bersama pasangan merupakan momen yang istimewa bagi kami.',
    '6. Saya dan pasangan menghabiskan waktu bersama dirumah.',
    '7. Dirumah, saya dan pasangan tidak hanya sekadar berbagi tempat yang sama, melainkan sebuah keluarga yang erat.',
    '8. Saya menikmati saat-saat liburan bersama pasangan.',
    '9. Saya menikmati bepergian bersama pasangan.',
    '10. Saya dan pasangan memiliki tujuan yang sama.',
    '11. Saya merasa bahwa perjalanan hidup saya dan pasangan akan tetap berjalan searah dan saling mendukung hingga masa depan.',
    '12. Saya dan pasangan memiliki pandangan yang serupa tentang privasi.',
    '13. Saya dan pasangan memiliki selera hiburan yang serupa.',
    '14. Saya dan pasangan memiliki tujuan dan prioritas yang sama terhadap orang-orang yang penting bagi kami (anak, teman dan lainnya).',
    '15. Saya dan pasangan memiliki impian  hidup yang sama dan selaras.',
    '16.  Saya dan pasangan memiliki pandangan yang sama tentang cinta. ',
    '17. Saya dan pasangan memiliki pandangan yang sama tentang bagaimana cara untuk bahagia.',
    '18.  Saya dan pasangan memiliki pandangan yang serupa tentang bagaimana seharusnya menjalani sebuah pernikahan. ',
    '19.  Saya dan pasangan memiliki pandangan yang sama mengenai pembagian peran dalam pernikahan.',
    '20. Saya dan pasangan memiliki nilai-nilai yang sama mengenai arti sebuah kepercayaan dalam hubungan.',
    '21. Saya tahu apa yang disukai pasangan saya',
    '22. Saya tahu bagaimana pasangan saya ingin dirawat ketika sedang sakit.',
    '23. Saya mengetahui makanan favorit pasangan saya.',
    '24. Saya mengetahui berbagai tekanan atau masalah yang sedang dihadapi pasangan saya.',
    '25. Saya memahami perasaan dan pikiran pasangan saya',
    '26. Saya tahu hal-hal yang membuat pasangan saya khawatir.',
    '27. Saya mengetahui sumber stres yang sedang dialami pasangan saya sehari-hari',
    '28. Saya mengetahui harapan dan keinginan pasangan saya.',
    '29. Saya mengenal pasangan saya dengan sangat baik.',
    '30. Saya mengetahui siapa saja teman-teman dan lingkungan pergaulan pasangan saya.',
    '31. Saat bertengkar dengan pasangan, saya merasa cenderung bersikap agresif. ',
    '32. Dalam pertengkaran, saya menggunakan kalimat "kamu selalu" atau "kamu tidak pernah".',
    '33. Ketika bertengkar, saya menyinggung tentang kepribadian negatif / hal negatif / kekurangan dalam diri pasangan saya.',
    '34. Ketika bertengkar, saya menggunakan kata kata menyakitkan / umpatan.',
    '35.  Ketika bertengkar, saya menghina pasangan saya. ',
    '36. Dalam pertengkaran, saya merendahkan pasangan.',
    '37. Dalam pertengkaran, saya dan pasangan tidak berakhir damai. ',
    '38. Saya tidak menyukai cara pasangan memulai sebuah topik pembicaraan.',
    '39. Pertengkaran terjadi secara tiba-tiba.',
    '40. Bahkan sebelum memahami apa yang terjadi, saya dan pasangan sudah mulai bertengkar',
    '41. Saat berbicara dengan pasangan tentang suatu masalah, ketenangan saya bisa tiba-tiba hilang.',
    '42.  Ketika bertengkar dengan pasangan, saya hanya menutup diri dan tidak mengatakan sepatah kata pun.',
    '43. Saya memilih membisu/menutup diri hanya demi meredam suasana.',
    '44. Saya berpikir bahwa meninggalkan rumah untuk sementara waktu akan menjadi hal yang baik.',
    '45. Daripada bertengkar dengan pasangan, saya lebih memilih untuk mendiamkannya/ tidak merespon lagi.',
    '46. Sekalipun saya berada di pihak yang benar,  saya memilih untuk tidak merespon pasangan saat terjadi pertengkaran. ',
    '47. Saat bertengkar dengan pasangan, saya memilih untuk menarik diri karena khawatir akan lepas kendali.',
    '48. Dalam pertengkaran dengan pasangan, saya merasa bahwa saya yang benar.',
    '49. Hal-hal yang dituduhkan pasangan kepada saya, tidak ada hubungan dengan saya.',
    '50. Sebenarnya bukan saya yang bersalah atas hal-hal yang dituduhkan kepada saya',
    '51. Saya bukan pihak yang bersalah atas masalah-masalah di rumah.',
    '52.Saya secara blakblakan memberi tahu kekurangan pasangan.',
    '53. Dalam pertengkaran, saya mengungkit kekurangan pasangan saya.',
    '54. Saya mengatakan kepada pasangan mengenai ketidakmampuan atau kelemahannya.',
]

KOLOM_FS = [
    '1. Seberapa tingkat stress keuangan Anda hari ini?',
    '2. Seberapa puas anda dengan kondisi keuangan Anda saat ini?',
    '3. Bagaimana perasaan Anda mengenai kondisi keuangan Anda saat ini?',
    '4. Seberapa sering Anda merasa khawatir tidak dapat memenuhi biaya hidup bulanan seperti biasanya?',
    '5. Seberapa yakin Anda bisa mendapatkan uang untuk mengatasi keadaan darurat keuangan yang membutuhkan sekitar Rp15.000.000?',
    '6. Seberapa sering Anda mengurungkan niat untuk pergi makan di luar, menonton film, atau melakukan sesuatu yang lain karena merasa tidak mampu?',
    '7. Seberapa sering Anda merasa hanya memiliki uang yang cukup untuk hidup, dan hidup dari gaji ke gaji?',
    '8. Seberapa stress Anda terhadap keuangan pribadi Anda secara umum?',
]

KOLOM_KDRT = [
    '1. Menurut saya seorang suami wajar menampar istrinya jika melakukan kesalahan.',
    '2. Menurut saya seorang suami perlu mengintai seluruh aktivitas istrinya dari jauh',
    '3. Menurut saya suami boleh menyadap alat komunikasi istrinya',
    '4. Menurut saya seorang suami tidak boleh menyalahkan istri atas segala permasalahan yang ada di dalam keluarga',
    '5. Menurut saya seorang suami pantas menghina istri karena menganggap tidak becus mengurus anak dan rumah tangga',
    '6. Menurut saya seorang suami tidak boleh mencekik istrinya saat terjadi pertengkaran',
    '7. Menurut saya jika seorang istri yang menyebabkan terjadinya pertengkaran, maka suami boleh mengancam dengan kepalan tangan untuk menghentikan perilaku istrinya',
    '8. Saya merasa seorang suami boleh berbicara dengan nada keras pada istri yang melakukan kesalahan',
    '9.Menurut saya seorang suami menghubungi istrinya setiap saat kerena rasa cemburu',
    '10. Menurut saya seorang suami boleh memarahi isstri dengan kata-kata kasar',
    '11. Saya merasa kesal jika melihat seorang suami membentak istri saat melakukan kesalahan',
    '12. Saya merasa kesal jika mendengar suami memaksa istrinya melakukan kekerasan seksual secara tidak wajar',
    '13. Saya merasa biasa saja saat melihat seorang suami mendorong istri sampai terjatuh',
    '14. Saya marah jika mengetahui suami memaksa istrinya untuk melakukan hubungan seksual walaupun dalam keadaan sakit ataupun lelah',
    '15. Saya merasa kesal jika mengetahui ada seorang suami yang melacak keberadaan istrinya menggunakan handphone setiap saat karena curiga',
    '16. Saya biasa saja saat melihat seorang suami yang merendahkan istrinya',
    '17. Saya marah seorang suami melakukan hal yang dapat melukai istrinya',
    '18. Saya akan marah jika mengetahui suami mencaci istrinya',
    '19. Saya merasa kesal jika suami memaksa istrinya melakukan hubungan seksual saat datang bulan',
    '20. Saya tersulut emosi ketika suami mengancam istri dengan kepalan',
    '21. Saya tidak suka saat melihat seorang suami yang memberikan komentar yang dapat melukai persaan istri',
    '22. Saya senang bila seorang suami memperdulikan istri yang dalam keadaan sakit',
    '23. Saya akan menasehati jika melihat seorang suami merusak barang-barang milik istrinya karena curiga',
    '24. Saya merasa biasa saja jika mengetahui suami memaksa melakukan kekerasan pada saat berhubungan intim hingga merasa kesakitan',
    '25. Tidak masalah bagi saya jika ada suami yang memaksa istri melakukan hubungan seksual sementara istri tidak menginginkannya',
    '26. Saya akan membiarkan jika melihat suami memarahi istri dengan kata-kata kasar',
    '27. Saya akan mencari bantuan jika seorang suami mengancamkan membunuh istrinya saat terjadi pertengkaran',
    '28. Saya akan menesehati jika seorang suami yang terus menerus menelpon istrinya karena curiga',
    '29. Saya akan membiarkan jika seorang suami menendang istrinya dengan keras terhadap istri',
    '30. Saya akan melapor ke polisi jika saya mengetahui adanya perlakuan seorang suami yang secara sengaja melukai istrinya',
    '31. Saya akan bertindak jika mengetahui suami memaksa istri untuk melakukan hubungan seksual dengan orang lain sebagai alasan untuk memenuhi kebutuhan ekonomi',
    '32. Saya akan menolong jika saya melihatseorang suami menganiaya istrinya',
    '33. Saya tidak akan membiarkan jika seorang suami menghubungi istrinya setiap saat karena rasa cemburu',
    '34. Saya tidak peduli jika seorang suami menghina istrinya dengan sebutan yang tidak menyenangkan seperti tolol, bodoh, jelek dan lain-lain',
    '35. Saya akan melapor jika seorang suami mengurung atau menyekap istrinya dalam ruangan',
    '36. Saya tidak akan membiarkan seorang suami jika suami mendorong kepala istrinya dengan keras',
    '37. Saya akan menasehati jika suami memaksa istri melakukan hubungan seksual',
    '38. Saya akan mencari bantuan jika suami melakukan suatu hal untuk menakuti istrinya',
    '39. Saya tidak akan menghalangi jika ada suami meraba-raba atau menyentuh bagian sensitif istri secara paksa hingga terluka',
    '40. Saya akan menasehati istri untuk menyampaikan pendapatnya apabila merasa tidak nyaman dalam berhubungan seksual',
]

# ==========================================================
# HASIL EVALUASI TUJUAN 1
# ==========================================================
EVAL_ROWS = [
    ("RF | SMOTE30",   0.8620, 0.6048, 0.0533, 0.9245, 0.6316, 0.2791, 0.3871, 0.3471, 0.9463, 0.6739, 0.7209, 0.6966),
    ("RF | SMOTE70",   0.8625, 0.5952, 0.0561, 0.9304, 0.6818, 0.3488, 0.4615, 0.3930, 0.9364, 0.6486, 0.5581, 0.6000),
    ("RF | no-SMOTE",  0.8491, 0.5914, 0.0520, 0.9245, 0.6667, 0.2326, 0.3448, 0.3224, 0.9344, 0.6000, 0.6977, 0.6452),
    ("RF | SMOTE50",   0.8596, 0.5910, 0.0554, 0.9324, 0.7143, 0.3488, 0.4688, 0.3748, 0.9364, 0.6341, 0.6047, 0.6190),
    ("RF | SMOTE100",  0.8578, 0.5890, 0.0585, 0.9264, 0.6250, 0.3488, 0.4478, 0.3864, 0.9284, 0.5854, 0.5581, 0.5714),
    ("ANN | SMOTE100", 0.8296, 0.5568, 0.0577, 0.9364, 0.6486, 0.5581, 0.6000, 0.5600, 0.9423, 0.7059, 0.5581, 0.6234),
    ("ANN | SMOTE70",  0.8278, 0.5299, 0.0583, 0.9344, 0.6389, 0.5349, 0.5823, 0.5573, 0.9384, 0.6765, 0.5349, 0.5974),
    ("ANN | SMOTE50",  0.8322, 0.5274, 0.0557, 0.9324, 0.6286, 0.5116, 0.5641, 0.6233, 0.9423, 0.7500, 0.4884, 0.5915),
    ("LR | no-SMOTE",  0.8184, 0.5024, 0.0609, 0.9284, 0.6129, 0.4419, 0.5135, 0.6323, 0.9344, 0.6786, 0.4419, 0.5352),
    ("LR | SMOTE50",   0.8030, 0.5009, 0.0834, 0.8887, 0.3818, 0.4884, 0.4286, 0.7727, 0.9225, 0.5556, 0.4651, 0.5063),
    ("LR | SMOTE70",   0.8036, 0.4998, 0.0868, 0.8867, 0.3833, 0.5349, 0.4466, 0.8218, 0.9225, 0.5556, 0.4651, 0.5063),
    ("LR | SMOTE30",   0.8097, 0.4978, 0.0761, 0.9026, 0.4375, 0.4884, 0.4615, 0.7889, 0.9304, 0.6333, 0.4419, 0.5205),
    ("LR | SMOTE100",  0.8012, 0.4919, 0.0930, 0.8827, 0.3710, 0.5349, 0.4381, 0.9053, 0.9264, 0.5938, 0.4419, 0.5067),
    ("ANN | SMOTE30",  0.8254, 0.4684, 0.0579, 0.9364, 0.6897, 0.4651, 0.5556, 0.4904, 0.9384, 0.7000, 0.4884, 0.5753),
    ("ANN | no-SMOTE", 0.7823, 0.3464, 0.0679, 0.9225, 0.6667, 0.1860, 0.2909, 0.2357, 0.8807, 0.3651, 0.5349, 0.4340),
    ("BNB | SMOTE100", 0.8207, 0.3227, 0.1835, 0.8012, 0.2522, 0.6744, 0.3671, 0.9075, 0.8310, 0.2900, 0.6744, 0.4056),
    ("BNB | SMOTE30",  0.8242, 0.3219, 0.1771, 0.8131, 0.2661, 0.6744, 0.3816, 1.0000, 0.8847, 0.3636, 0.4651, 0.4082),
    ("BNB | SMOTE70",  0.8233, 0.3070, 0.1774, 0.8131, 0.2661, 0.6744, 0.3816, 1.0000, 0.8807, 0.3607, 0.5116, 0.4231),
    ("BNB | no-SMOTE", 0.8109, 0.3063, 0.1787, 0.8131, 0.2661, 0.6744, 0.3816, 1.0000, 0.8827, 0.3571, 0.4651, 0.4040),
    ("BNB | SMOTE50",  0.8191, 0.3022, 0.1764, 0.8131, 0.2661, 0.6744, 0.3816, 0.9998, 0.8688, 0.3380, 0.5581, 0.4211),
]
EVAL_COLS = ["skenario", "roc_auc", "pr_auc", "brier",
             "acc@0.5", "prec@0.5", "rec@0.5", "f1@0.5",
             "thr_best", "acc_best", "prec_best", "rec_best", "f1_best"]

# ==========================================================
# HASIL TUJUAN 2 (16 FITUR) - Hardcoded dari analisis
# ==========================================================
T2_HASIL = pd.DataFrame([
    {"Fitur": "Shared_Meaning", "beta": -0.9901, "ci_low": -2.2513, "ci_high": -0.1966,
     "importance": 0.2287, "kategori": "PROTEKTIF"},
    {"Fitur": "Neg_Conflict", "beta": 0.8591, "ci_low": 0.3516, "ci_high": 1.6487,
     "importance": 0.1759, "kategori": "RISIKO"},
    {"Fitur": "KDRT_Psikologis_Fav", "beta": 0.8974, "ci_low": 0.3522, "ci_high": 1.8373,
     "importance": 0.1198, "kategori": "RISIKO"},
    {"Fitur": "Pendapatan", "beta": -0.5564, "ci_low": -1.7225, "ci_high": -0.0022,
     "importance": 0.0562, "kategori": "PROTEKTIF"},
    {"Fitur": "Financial_WellBeing", "beta": 0.4653, "ci_low": -0.1755, "ci_high": 1.5849,
     "importance": 0.0578, "kategori": "NETRAL"},
    {"Fitur": "KDRT_Seksual_Unf", "beta": 0.5953, "ci_low": -0.1424, "ci_high": 1.4557,
     "importance": 0.0279, "kategori": "NETRAL"},
    {"Fitur": "KDRT_Psikologis_Unf", "beta": 0.4853, "ci_low": -0.2768, "ci_high": 1.5779,
     "importance": 0.0309, "kategori": "NETRAL"},
    {"Fitur": "Stonewalling", "beta": -0.2024, "ci_low": -0.8119, "ci_high": 0.2454,
     "importance": 0.0196, "kategori": "NETRAL"},
    {"Fitur": "Defensiveness", "beta": -0.1659, "ci_low": -0.8336, "ci_high": 0.3562,
     "importance": 0.0084, "kategori": "NETRAL"},
    {"Fitur": "KDRT_Fisik_Unf", "beta": 0.1208, "ci_low": -0.6215, "ci_high": 0.9918,
     "importance": 0.0031, "kategori": "NETRAL"},
    {"Fitur": "KDRT_Memata_matai_Fav", "beta": -0.0558, "ci_low": -0.8599, "ci_high": 0.5309,
     "importance": 0.0033, "kategori": "NETRAL"},
    {"Fitur": "Love_Maps", "beta": -0.0923, "ci_low": -0.8984, "ci_high": 0.7369,
     "importance": 0.0005, "kategori": "NETRAL"},
    {"Fitur": "Pendidikan", "beta": 0.1336, "ci_low": -0.4727, "ci_high": 0.6791,
     "importance": -0.0018, "kategori": "NETRAL"},
    {"Fitur": "KDRT_Fisik_Fav", "beta": -0.2820, "ci_low": -0.9101, "ci_high": 0.3535,
     "importance": -0.0038, "kategori": "NETRAL"},
    {"Fitur": "KDRT_Seksual_Fav", "beta": -0.1326, "ci_low": -0.6849, "ci_high": 0.4260,
     "importance": -0.0017, "kategori": "NETRAL"},
    {"Fitur": "KDRT_Memata_matai_Unf", "beta": 0.2293, "ci_low": -0.5506, "ci_high": 1.0746,
     "importance": -0.0017, "kategori": "NETRAL"},
])
T2_HASIL["skor_dominan"] = T2_HASIL["beta"].abs() * np.sqrt(T2_HASIL["importance"].clip(lower=0))

T2_STABILITAS = pd.DataFrame([
    {"Fitur": "Pendapatan", "beta_full": -0.556, "beta_cv": -0.598, "sd": 0.229, "proporsi": 1.000, "Status": "Stabil"},
    {"Fitur": "Shared_Meaning", "beta_full": -0.990, "beta_cv": -1.020, "sd": 0.310, "proporsi": 1.000, "Status": "Stabil"},
    {"Fitur": "Neg_Conflict", "beta_full": 0.859, "beta_cv": 0.882, "sd": 0.167, "proporsi": 1.000, "Status": "Stabil"},
    {"Fitur": "KDRT_Psikologis_Fav", "beta_full": 0.897, "beta_cv": 0.923, "sd": 0.217, "proporsi": 1.000, "Status": "Stabil"},
    {"Fitur": "KDRT_Seksual_Unf", "beta_full": 0.595, "beta_cv": 0.609, "sd": 0.214, "proporsi": 1.000, "Status": "Stabil"},
    {"Fitur": "Financial_WellBeing", "beta_full": 0.465, "beta_cv": 0.490, "sd": 0.229, "proporsi": 1.000, "Status": "Stabil"},
    {"Fitur": "KDRT_Psikologis_Unf", "beta_full": 0.485, "beta_cv": 0.503, "sd": 0.254, "proporsi": 0.980, "Status": "Stabil"},
    {"Fitur": "KDRT_Fisik_Fav", "beta_full": -0.282, "beta_cv": -0.274, "sd": 0.157, "proporsi": 0.940, "Status": "Stabil"},
    {"Fitur": "Stonewalling", "beta_full": -0.202, "beta_cv": -0.204, "sd": 0.144, "proporsi": 0.900, "Status": "Stabil"},
    {"Fitur": "KDRT_Memata_matai_Unf", "beta_full": 0.229, "beta_cv": 0.227, "sd": 0.194, "proporsi": 0.900, "Status": "Stabil"},
    {"Fitur": "KDRT_Fisik_Unf", "beta_full": 0.121, "beta_cv": 0.142, "sd": 0.217, "proporsi": 0.820, "Status": "Marginal"},
    {"Fitur": "Pendidikan", "beta_full": 0.134, "beta_cv": 0.131, "sd": 0.158, "proporsi": 0.800, "Status": "Marginal"},
    {"Fitur": "Defensiveness", "beta_full": -0.166, "beta_cv": -0.169, "sd": 0.157, "proporsi": 0.800, "Status": "Marginal"},
    {"Fitur": "KDRT_Seksual_Fav", "beta_full": -0.133, "beta_cv": -0.131, "sd": 0.133, "proporsi": 0.800, "Status": "Marginal"},
    {"Fitur": "Love_Maps", "beta_full": -0.092, "beta_cv": -0.096, "sd": 0.168, "proporsi": 0.720, "Status": "Tidak Stabil"},
    {"Fitur": "KDRT_Memata_matai_Fav", "beta_full": -0.056, "beta_cv": -0.064, "sd": 0.142, "proporsi": 0.640, "Status": "Tidak Stabil"},
])

T2_VIF = pd.DataFrame([
    {"Fitur": "KDRT_Psikologis_Unf", "VIF": 3.165, "Status": "Aman"},
    {"Fitur": "KDRT_Seksual_Unf", "VIF": 3.006, "Status": "Aman"},
    {"Fitur": "KDRT_Fisik_Unf", "VIF": 2.842, "Status": "Aman"},
    {"Fitur": "Shared_Meaning", "VIF": 2.820, "Status": "Aman"},
    {"Fitur": "Neg_Conflict", "VIF": 2.604, "Status": "Aman"},
    {"Fitur": "Love_Maps", "VIF": 2.561, "Status": "Aman"},
    {"Fitur": "KDRT_Psikologis_Fav", "VIF": 2.513, "Status": "Aman"},
    {"Fitur": "KDRT_Memata_matai_Unf", "VIF": 2.235, "Status": "Aman"},
    {"Fitur": "Defensiveness", "VIF": 2.158, "Status": "Aman"},
    {"Fitur": "KDRT_Fisik_Fav", "VIF": 1.943, "Status": "Aman"},
    {"Fitur": "KDRT_Seksual_Fav", "VIF": 1.634, "Status": "Aman"},
    {"Fitur": "KDRT_Memata_matai_Fav", "VIF": 1.458, "Status": "Aman"},
    {"Fitur": "Stonewalling", "VIF": 1.441, "Status": "Aman"},
    {"Fitur": "Pendapatan", "VIF": 1.432, "Status": "Aman"},
    {"Fitur": "Financial_WellBeing", "VIF": 1.355, "Status": "Aman"},
    {"Fitur": "Pendidikan", "VIF": 1.302, "Status": "Aman"},
])

# ==========================================================
# HASIL TUJUAN 3 (6 HIPOTESIS INTERAKSI)
# ==========================================================
T3_HASIL = pd.DataFrame([
    {"Hipotesis": "Shared x Pendapatan", "Kelompok": "Sosial ekonomi",
     "beta": -0.243, "ci_low": -0.680, "ci_high": 0.194,
     "p_Wald": 0.276, "p_LRT": 0.266, "p_FDR": 0.532, "Status": "Tidak Signifikan"},
    {"Hipotesis": "Konflik x Pendidikan", "Kelompok": "Sosial ekonomi",
     "beta": -0.087, "ci_low": -0.390, "ci_high": 0.215,
     "p_Wald": 0.571, "p_LRT": 0.572, "p_FDR": 0.687, "Status": "Tidak Signifikan"},
    {"Hipotesis": "Konflik x Stress", "Kelompok": "Keuangan",
     "beta": -0.214, "ci_low": -0.504, "ci_high": 0.075,
     "p_Wald": 0.147, "p_LRT": 0.136, "p_FDR": 0.408, "Status": "Tidak Signifikan"},
    {"Hipotesis": "Shared x Stress", "Kelompok": "Keuangan",
     "beta": 0.341, "ci_low": 0.029, "ci_high": 0.653,
     "p_Wald": 0.032, "p_LRT": 0.026, "p_FDR": 0.155, "Status": "Signifikan Raw"},
    {"Hipotesis": "Konflik x Permisif", "Kelompok": "Sikap KDRT",
     "beta": -0.016, "ci_low": -0.333, "ci_high": 0.302,
     "p_Wald": 0.923, "p_LRT": 0.923, "p_FDR": 0.923, "Status": "Tidak Signifikan"},
    {"Hipotesis": "Defensif x Permisif", "Kelompok": "Sikap KDRT",
     "beta": -0.125, "ci_low": -0.421, "ci_high": 0.170,
     "p_Wald": 0.406, "p_LRT": 0.405, "p_FDR": 0.607, "Status": "Tidak Signifikan"},
])

T3_STABILITAS = pd.DataFrame([
    {"Hipotesis": "Shared x Pendapatan", "beta_full": -0.243, "beta_cv": -0.239, "sd": 0.105, "proporsi": 0.96, "Status": "Stabil"},
    {"Hipotesis": "Konflik x Pendidikan", "beta_full": -0.087, "beta_cv": -0.085, "sd": 0.098, "proporsi": 0.80, "Status": "Marginal"},
    {"Hipotesis": "Konflik x Stress", "beta_full": -0.214, "beta_cv": -0.218, "sd": 0.087, "proporsi": 1.00, "Status": "Stabil"},
    {"Hipotesis": "Shared x Stress", "beta_full": 0.341, "beta_cv": 0.351, "sd": 0.103, "proporsi": 1.00, "Status": "Stabil"},
    {"Hipotesis": "Konflik x Permisif", "beta_full": -0.016, "beta_cv": -0.021, "sd": 0.117, "proporsi": 0.54, "Status": "Tidak Stabil"},
    {"Hipotesis": "Defensif x Permisif", "beta_full": -0.125, "beta_cv": -0.124, "sd": 0.072, "proporsi": 0.96, "Status": "Stabil"},
])

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def clean_question(col_name):
    s = str(col_name).strip()
    s = re.sub(r'^\s*\d+[\.\)]\s*', '', s)
    return s.strip()

def decode(s, dec):
    if s.dtype == object:
        return s
    return s.map(dec).fillna(s.astype(str))

def odds_interpretation(beta):
    """Narasi interpretasi beta ke odds ratio."""
    or_val = np.exp(beta)
    if beta > 0:
        return f"setiap kenaikan 1 SD meningkatkan odds perceraian sekitar {or_val:.2f} kali lipat"
    elif beta < 0:
        return f"setiap kenaikan 1 SD menurunkan odds perceraian menjadi sekitar {or_val:.2f} kali"
    else:
        return "tidak ada perubahan odds"

def classify_lean(beta, ci_low, ci_high):
    """Klasifikasi kecenderungan arah ketika CI melintasi 0."""
    if ci_low > 0:
        return "Arah positif konsisten"
    if ci_high < 0:
        return "Arah negatif konsisten"
    # CI melintasi 0
    if beta > 0.2:
        return "Condong ke arah risiko, namun tidak signifikan"
    elif beta < -0.2:
        return "Condong ke arah protektif, namun tidak signifikan"
    else:
        return "Arah tidak jelas (mendekati nol)"

def render_narasi_t2(row):
    """Bangun narasi interpretasi untuk satu faktor Tujuan 2."""
    beta = row["beta"]
    ci_lo = row["ci_low"]
    ci_hi = row["ci_high"]
    kat = row["kategori"]
    fitur = row["Fitur"]

    if kat == "RISIKO":
        narasi = f"Setiap kenaikan 1 SD pada {fitur} meningkatkan odds perceraian sekitar {np.exp(beta):.2f} kali lipat. Interval kepercayaan tidak melintasi nol, sehingga arah risiko konsisten dan dapat disimpulkan."
    elif kat == "PROTEKTIF":
        narasi = f"Setiap kenaikan 1 SD pada {fitur} menurunkan odds perceraian menjadi sekitar {np.exp(beta):.2f} kali (turun sekitar {(1-np.exp(beta))*100:.0f} persen). Interval kepercayaan tidak melintasi nol, sehingga efek protektif dapat disimpulkan."
    else:
        lean = classify_lean(beta, ci_lo, ci_hi)
        narasi = f"Interval kepercayaan melintasi nol, sehingga arah tidak dapat dipastikan. {lean}."

    return narasi

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.header("Konfigurasi File")
    model_path = st.text_input("Model (.joblib)", DEFAULT_MODEL)
    oof_path   = st.text_input("OOF probabilities (.csv)", DEFAULT_OOF)
    data_path  = st.text_input("Data bersih (.csv)", DEFAULT_DATA)

    st.divider()
    st.markdown("### Legend Decoding")
    with st.expander("Pendidikan (ordinal)"):
        st.write(DECODE_PENDIDIKAN)
    with st.expander("DPS (0-4)"):
        st.write(DECODE_DPS)
    with st.expander("KDRT (1-4)"):
        st.write(DECODE_KDRT)
    with st.expander("Status Pernikahan"):
        st.write(DECODE_NIKAH)

    st.divider()
    st.caption("Dashboard Penelitian: Tujuan 1, 2, dan 3")

# ==========================================================
# HEADER
# ==========================================================
st.title("Dashboard Penelitian Perceraian")
st.caption("Analisis tiga tujuan: prediksi status pernikahan, identifikasi faktor dominan, dan analisis interaksi moderasi.")

# ==========================================================
# MAIN TABS
# ==========================================================
main_tab1, main_tab2, main_tab3 = st.tabs([
    "Tujuan 1 - Prediksi",
    "Tujuan 2 - Faktor Dominan",
    "Tujuan 3 - Analisis Interaksi",
])

# ==========================================================
# TUJUAN 1 - PREDIKSI
# ==========================================================
with main_tab1:
    st.header("Tujuan 1: Prediksi Status Pernikahan")
    st.caption("Model Random Forest dengan SMOTE 30 persen untuk mengklasifikasikan status pernikahan.")

    missing = [p for p in [model_path, oof_path, data_path] if not Path(p).exists()]
    if missing:
        st.error("File tidak ditemukan:\n" + "\n".join(f"- {m}" for m in missing))
        st.stop()

    try:
        artifact = joblib.load(model_path)
        oof      = pd.read_csv(oof_path).reset_index(drop=True)
        raw      = pd.read_csv(data_path).reset_index(drop=True)
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        st.stop()

    unnamed_cols = [c for c in raw.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        raw = raw.drop(columns=unnamed_cols)
    oof_unnamed = [c for c in oof.columns if str(c).startswith("Unnamed")]
    if oof_unnamed:
        oof = oof.drop(columns=oof_unnamed)
    if len(oof) != len(raw):
        st.error(f"Panjang tidak cocok: OOF={len(oof)} vs raw={len(raw)}")
        st.stop()

    metrics       = artifact.get("metrics_oof", {})
    thr_best      = artifact.get("threshold_best", 0.5)
    tinfo         = artifact.get("training_info", {})
    model         = artifact["model"]
    feature_names = artifact.get("feature_names", [])

    raw_cols = raw.columns.tolist()
    pendidikan_col = "Pendidikan Terakhir" if "Pendidikan Terakhir" in raw_cols else raw_cols[0]
    income_col = None
    for c in raw_cols:
        if "Pendapatan" in str(c) or "pendapatan" in str(c):
            income_col = c
            break
    if income_col is None:
        income_col = raw_cols[-1]

    used_cols = {pendidikan_col, income_col, "Status Pernikahan"}
    dps_cols = [c for c in raw_cols if c not in used_cols][:54]
    used_cols.update(dps_cols)
    fs_cols = [c for c in raw_cols if c not in used_cols][:8]
    used_cols.update(fs_cols)
    kdrt_cols = [c for c in raw_cols if c not in used_cols][:40]
    fs_min = int(raw[fs_cols].min().min())
    fs_max = int(raw[fs_cols].max().max())
    actual_feature_count = len([c for c in raw_cols if c != "Status Pernikahan"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("N Baris", f"{tinfo.get('n_samples', len(raw))}")
    c2.metric("N Fitur", f"{tinfo.get('n_features', '?')}")
    c3.metric("Positif (1)", f"{tinfo.get('n_positif', int(oof['y_true'].sum()))}")
    c4.metric("PR-AUC (OOF)", f"{metrics.get('pr_auc', 0):.4f}")
    c5.metric("ROC-AUC (OOF)", f"{metrics.get('roc_auc', 0):.4f}")
    c6.metric("Threshold Optimal", f"{thr_best:.4f}")
    st.divider()

    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "Metodologi",
        "Data & Distribusi",
        "Fold & SMOTE",
        "Hasil Evaluasi",
        "Prediksi Manual",
    ])

    with sub1:
        st.subheader("Metodologi Tujuan 1")
        st.markdown("""
        **Tujuan**: memprediksi status pernikahan (Menikah vs Cerai) berdasarkan 104 item kuesioner.

        **Pendekatan**:
        1. Data 503 responden dengan 104 item kuesioner sebagai fitur.
        2. Target biner: 0 = Menikah (460), 1 = Cerai (43). Rasio imbalance 10.7 banding 1.
        3. Untuk menangani ketidakseimbangan kelas, digunakan teknik SMOTE pada data train.
        4. Empat model diuji: Random Forest, Artificial Neural Network, Logistic Regression, dan Bernoulli Naive Bayes.
        5. Setiap model diuji dengan 5 level SMOTE: tanpa SMOTE, 30 persen, 50 persen, 70 persen, dan 100 persen.
        6. Evaluasi menggunakan 5-fold Stratified Cross-Validation dengan prediksi Out-Of-Fold.
        7. Metrik utama: PR-AUC karena kelas sangat tidak seimbang. ROC-AUC sebagai pendukung.

        **Mengapa SMOTE hanya di train**:
        SMOTE mensintesis sampel kelas minoritas. Jika diterapkan ke test, hasil evaluasi menjadi tidak jujur karena model akan diuji pada data sintetis. Karena itu SMOTE hanya diterapkan pada fold train, dan test tetap original.

        **Mengapa threshold 0.5 tidak dipakai**:
        Karena data tidak seimbang, threshold optimal dicari berdasarkan F1-maximum. Threshold ini lebih rendah dari 0.5 untuk menyeimbangkan precision dan recall.
        """)

    with sub2:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Distribusi Target")
            dist = oof["y_true"].value_counts().sort_index()
            dist_df = pd.DataFrame({
                "Kelas": [DECODE_NIKAH.get(i, str(i)) for i in dist.index],
                "Jumlah": dist.values,
            })
            dist_df["Proporsi (%)"] = (dist_df["Jumlah"] / dist_df["Jumlah"].sum() * 100).round(2)
            st.dataframe(dist_df, hide_index=True, use_container_width=True)

            fig = px.pie(dist_df, names="Kelas", values="Jumlah",
                         color="Kelas", hole=0.5,
                         color_discrete_map={"Menikah": "#10b981",
                                             "Cerai Hidup / Pernah Bercerai": "#ef4444"})
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Statistik Ringkas")
            imbalance = dist.iloc[0] / dist.iloc[1] if len(dist) > 1 else np.nan
            st.markdown(f"""
            - Total baris: {len(raw)}
            - Fitur untuk model: {actual_feature_count}
            - Imbalance ratio: 1 : {imbalance:.1f}
            - SMOTE ratio dipakai: {tinfo.get('smote_ratio', 0.30)}
            - Threshold optimal (F1-max): {thr_best:.4f}
            """)
            st.subheader("Preview Data")
            st.dataframe(raw.iloc[:5, :8], use_container_width=True)

    with sub3:
        st.subheader("Detail Fold - StratifiedKFold(5)")
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_rows = []
        for i, (tr, te) in enumerate(cv.split(raw, oof["y_true"]), start=1):
            y_tr = oof["y_true"].iloc[tr]
            y_te = oof["y_true"].iloc[te]
            fold_rows.append({"Fold": i, "Train n": len(tr),
                              "Train 0": int((y_tr == 0).sum()),
                              "Train 1": int((y_tr == 1).sum()),
                              "Test n": len(te),
                              "Test 0": int((y_te == 0).sum()),
                              "Test 1": int((y_te == 1).sum())})
        st.dataframe(pd.DataFrame(fold_rows), hide_index=True, use_container_width=True)
        st.info("StratifiedKFold menjaga proporsi kelas di setiap fold. Setiap baris muncul tepat 1 kali sebagai test dan 4 kali sebagai train.")

        st.divider()
        st.subheader("Simulasi SMOTE di Fold 1")
        sim_rows = [
            {"Skenario": "Tanpa SMOTE", "Train n": 402, "Train 0": 368, "Train 1": 34, "Proporsi 1": 0.085},
            {"Skenario": "SMOTE30", "Train n": 478, "Train 0": 368, "Train 1": 110, "Proporsi 1": 0.230},
            {"Skenario": "SMOTE50", "Train n": 552, "Train 0": 368, "Train 1": 184, "Proporsi 1": 0.333},
            {"Skenario": "SMOTE70", "Train n": 625, "Train 0": 368, "Train 1": 257, "Proporsi 1": 0.411},
            {"Skenario": "SMOTE100", "Train n": 736, "Train 0": 368, "Train 1": 368, "Proporsi 1": 0.500},
        ]
        st.dataframe(pd.DataFrame(sim_rows), hide_index=True, use_container_width=True)

    with sub4:
        df_eval = pd.DataFrame(EVAL_ROWS, columns=EVAL_COLS).set_index("skenario")
        st.subheader("Tabel Lengkap (diurutkan by PR-AUC)")
        st.dataframe(
            df_eval.sort_values("pr_auc", ascending=False)
                   .style.format("{:.4f}")
                   .background_gradient(subset=["pr_auc", "roc_auc", "f1_best"], cmap="Greens"),
            use_container_width=True, height=520,
        )
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pivot: PR-AUC per Model dan SMOTE")
            df_r = df_eval.reset_index()
            split_df = df_r["skenario"].str.split(" | ", n=1, regex=False, expand=True)
            df_r["model"] = split_df[0]
            df_r["skema"] = split_df[1]
            pivot = df_r.pivot(index="model", columns="skema", values="pr_auc")
            pivot = pivot[["no-SMOTE", "SMOTE30", "SMOTE50", "SMOTE70", "SMOTE100"]]
            st.dataframe(pivot.style.format("{:.4f}").background_gradient(cmap="Greens"), use_container_width=True)

            fig = px.line(pivot.T.reset_index().melt(id_vars="skema"),
                          x="skema", y="value", color="model", markers=True,
                          labels={"value": "PR-AUC", "skema": "SMOTE Ratio"})
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top 5 Model")
            top5 = df_eval.sort_values("pr_auc", ascending=False).head(5)
            st.dataframe(top5[["pr_auc", "roc_auc", "f1_best", "rec_best", "prec_best"]]
                         .style.format("{:.4f}").background_gradient(cmap="Greens"),
                         use_container_width=True)
            st.success(f"Pemenang: RF dengan SMOTE30. PR-AUC = {top5.iloc[0]['pr_auc']:.4f}, F1 = {top5.iloc[0]['f1_best']:.4f}.")

    with sub5:
        st.subheader("Prediksi Manual")
        st.info("Isi semua pertanyaan. Jawaban tinggal pilih, hanya pendapatan yang diisi angka.")

        with st.form("inference_form"):
            st.markdown("### Profil Responden")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                pendidikan_input = st.selectbox(
                    "Pendidikan Terakhir",
                    options=list(DECODE_PENDIDIKAN.keys()),
                    format_func=lambda x: f"{x} - {DECODE_PENDIDIKAN[x]}",
                    index=1,
                )
            with c_p2:
                income_input = st.number_input("Pendapatan per Bulan (Rp)",
                                                min_value=0, max_value=1_000_000_000,
                                                value=5_000_000, step=500_000, format="%d")

            st.divider()
            tab_dps, tab_fs, tab_kdrt = st.tabs([
                f"DPS ({len(dps_cols)} item)",
                f"FS ({len(fs_cols)} item)",
                f"KDRT ({len(kdrt_cols)} item)",
            ])

            dps_result = {}
            with tab_dps:
                st.caption("Skala: 0 = Tidak pernah, 4 = Selalu")
                for aspek, items in DPS_GROUPS.items():
                    with st.expander(f"{aspek} - {len(items)} item"):
                        for i in items:
                            q = clean_question(dps_cols[i - 1])
                            dps_result[i] = st.radio(f"{i}. {q}",
                                                      options=[0, 1, 2, 3, 4],
                                                      index=2, horizontal=True,
                                                      key=f"infer_dps_{i}")

            fs_result = {}
            with tab_fs:
                st.caption(f"Skala: {fs_min} (negatif) - {fs_max} (positif)")
                for i in range(1, 9):
                    q = clean_question(fs_cols[i - 1])
                    fs_result[i] = st.slider(f"{i}. {q}",
                                              min_value=fs_min, max_value=fs_max,
                                              value=5, step=1, key=f"infer_fs_{i}")

            kdrt_result = {}
            with tab_kdrt:
                st.caption("Skala: 1 = Sangat Tidak Setuju, 4 = Sangat Setuju")
                for aspek, items in KDRT_GROUPS.items():
                    with st.expander(f"{aspek} - {len(items)} item"):
                        for i in items:
                            q = clean_question(kdrt_cols[i - 1])
                            kdrt_result[i] = st.radio(f"{i}. {q}",
                                                       options=[1, 2, 3, 4],
                                                       index=1, horizontal=True,
                                                       key=f"infer_kdrt_{i}")

            st.divider()
            submitted = st.form_submit_button("Prediksi Status Pernikahan",
                                                type="primary", use_container_width=True)

        if submitted:
            feature_cols = [c for c in raw_cols if c != "Status Pernikahan"]
            values = []
            for c in feature_cols:
                if c == pendidikan_col:
                    values.append(pendidikan_input)
                elif c == income_col:
                    values.append(income_input)
                elif c in dps_cols:
                    values.append(dps_result.get(dps_cols.index(c) + 1, 2))
                elif c in fs_cols:
                    values.append(fs_result.get(fs_cols.index(c) + 1, 5))
                elif c in kdrt_cols:
                    values.append(kdrt_result.get(kdrt_cols.index(c) + 1, 2))
                else:
                    values.append(0)

            X_pred = np.array(values, dtype=float).reshape(1, -1)
            if X_pred.shape[1] != len(feature_names):
                st.error(f"Panjang fitur tidak cocok: model expects {len(feature_names)}, input {X_pred.shape[1]}")
            else:
                prob = float(model.predict_proba(X_pred)[0, 1])
                pred = int(prob >= thr_best)
                st.divider()
                st.markdown("## Hasil Prediksi")
          

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Probabilitas Cerai"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ef4444" if pred == 1 else "#10b981"},
                        "threshold": {"line": {"color": "black", "width": 3},
                                       "value": thr_best * 100},
                    },
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                if prob < 0.20:
                    st.success("Kategori Risiko: Sangat Rendah")
                elif prob < 0.40:
                    st.success("Kategori Risiko: Rendah")
                elif prob < 0.60:
                    st.warning("Kategori Risiko: Sedang")
                elif prob < 0.80:
                    st.warning("Kategori Risiko: Tinggi")
                else:
                    st.error("Kategori Risiko: Sangat Tinggi")


# ==========================================================
# TUJUAN 2 - FAKTOR DOMINAN
# ==========================================================
with main_tab2:
    st.header("Tujuan 2: Faktor Risiko dan Protektif Dominan")
    st.caption("Regresi logistik dengan 16 fitur, regularisasi L2, dan bootstrap 1000 iterasi.")

    sub1, sub2, sub3, sub4, sub5, sub6, sub7 = st.tabs([
        "Metodologi",
        "Uji Asumsi Awal",
        "Struktur 16 Fitur",
        "Hasil Regresi",
        "Interpretasi Klasifikasi",
        "Stabilitas CV",
        "Korelasi",
    ])

    with sub1:
        st.subheader("Metodologi Tujuan 2")
        st.markdown("""
        **Tujuan**: mengidentifikasi faktor risiko (yang meningkatkan probabilitas perceraian) dan faktor protektif (yang menurunkan probabilitas perceraian) yang dominan.

        **Pendekatan**:
        1. Dari 104 item kuesioner, dibentuk 16 subskala berdasarkan blueprint instrumen (2 demografis, 5 dari DPS, 1 dari IFDFW, 8 dari KDRT).
        2. Setiap subskala dihitung sebagai rata-rata itemnya.
        3. Model regresi logistik digunakan karena target bersifat biner.
        4. Semua fitur distandardisasi (z-score) agar koefisien bisa dibandingkan secara langsung.
        5. Regularisasi L2 dengan C=1.0 diterapkan untuk menahan koefisien saat EPV rendah.
        6. Class weight balanced untuk mengatasi ketidakseimbangan data.
        7. Bootstrap 1000 iterasi untuk interval kepercayaan yang lebih stabil pada sampel kecil.
        8. Permutation importance untuk mengukur kontribusi tiap fitur terhadap performa model.

        **Mengapa regresi logistik**:
        - Target biner (Menikah vs Cerai).
        - Koefisien dapat langsung diinterpretasi arah dan besarnya.
        - Hasilnya interpretable untuk penelitian eksploratif.

        **Klasifikasi tiga tingkat**:
        - RISIKO: koefisien positif dan interval kepercayaan tidak melintasi nol.
        - PROTEKTIF: koefisien negatif dan interval kepercayaan tidak melintasi nol.
        - NETRAL: interval kepercayaan melintasi nol, sehingga arah tidak dapat dipastikan.

        **Mengapa 16 fitur**:
        Meski EPV rendah (2.53), 16 fitur tetap dianalisis untuk eksplorasi detail tiap dimensi KDRT (fisik, seksual, memata-matai, psikologis). Untuk klaim utama, versi agregat (10 fitur) lebih stabil.
        """)

    with sub2:
        st.subheader("Uji Asumsi Awal Data")
        st.markdown("""
        Sebelum analisis, dilakukan pemeriksaan awal terhadap kualitas data.
        """)

        asumsi_t2 = pd.DataFrame([
            {"No": 1, "Uji": "Missing values", "Hasil": "0 nilai hilang", "Status": "Baik",
             "Penjelasan": "Data lengkap, tidak perlu imputasi"},
            {"No": 2, "Uji": "Distribusi target", "Hasil": "460 menikah vs 43 cerai (10.7:1)", "Status": "Tidak seimbang",
             "Penjelasan": "Ditangani dengan class_weight balanced dan bootstrap"},
            {"No": 3, "Uji": "Multikolinearitas (VIF)", "Hasil": "Max = 3.16", "Status": "Baik",
             "Penjelasan": "Semua VIF di bawah 5, tidak ada multikolinearitas bermasalah"},
            {"No": 4, "Uji": "Korelasi tinggi (> 0.7)", "Hasil": "4 pasang", "Status": "Perlu diperhatikan",
             "Penjelasan": "Tidak di-drop karena tiap subskala mengukur dimensi berbeda"},
            {"No": 5, "Uji": "Events Per Variable (EPV)", "Hasil": "43 / 17 = 2.53", "Status": "Rendah",
             "Penjelasan": "Ditangani dengan regularisasi L2 dan bootstrap CI"},
        ])
        st.dataframe(asumsi_t2, hide_index=True, use_container_width=True)

        st.markdown("""
        **Istilah penting**:

        **VIF (Variance Inflation Factor)** mengukur seberapa besar varians koefisien regresi meningkat akibat korelasi antar prediktor. Nilai VIF di bawah 5 dianggap aman, di atas 10 mengindikasikan multikolinearitas bermasalah.

        **EPV (Events Per Variable)** adalah rasio antara jumlah kejadian (kasus positif, yaitu 43 orang yang pernah bercerai) dengan jumlah parameter yang diestimasi (17, yaitu 16 fitur ditambah intersep). Nilai EPV minimal 10 dianggap cukup untuk regresi logistik yang stabil (Peduzzi et al., 1996). Nilai EPV 2.53 berada jauh di bawah ambang ideal tersebut.

        **Mengapa 4 pasang korelasi tinggi tidak di-drop**:
        Shared Meaning dan Love Maps berkorelasi 0.757, namun keduanya mengukur dimensi berbeda (makna bersama vs pengetahuan tentang pasangan). Empat subskala KDRT Unfavorable berkorelasi 0.6 sampai 0.73, namun mewakili jenis kekerasan berbeda (fisik, seksual, memata-matai, psikologis). Korelasi tinggi saja tidak cukup alasan untuk drop jika secara konseptual berbeda.
        """)

    with sub3:
        st.subheader("Struktur 16 Fitur")
        struktur = pd.DataFrame([
            {"No": 1, "Instrumen": "Demografis", "Subskala": "Pendidikan", "Jumlah Item": "-", "Skala": "1-3", "Arah Teori": "Netral"},
            {"No": 2, "Instrumen": "Demografis", "Subskala": "Pendapatan", "Jumlah Item": "-", "Skala": "Rupiah", "Arah Teori": "Protektif"},
            {"No": 3, "Instrumen": "DPS", "Subskala": "Shared Meaning & Forgiveness", "Jumlah Item": 20, "Skala": "0-4", "Arah Teori": "Protektif"},
            {"No": 4, "Instrumen": "DPS", "Subskala": "Love Maps", "Jumlah Item": 10, "Skala": "0-4", "Arah Teori": "Protektif"},
            {"No": 5, "Instrumen": "DPS", "Subskala": "Negative Conflict Behaviors", "Jumlah Item": 11, "Skala": "0-4", "Arah Teori": "Risiko"},
            {"No": 6, "Instrumen": "DPS", "Subskala": "Stonewalling", "Jumlah Item": 6, "Skala": "0-4", "Arah Teori": "Risiko"},
            {"No": 7, "Instrumen": "DPS", "Subskala": "Defensiveness", "Jumlah Item": 7, "Skala": "0-4", "Arah Teori": "Risiko"},
            {"No": 8, "Instrumen": "IFDFW", "Subskala": "Financial Well-Being", "Jumlah Item": 8, "Skala": "1-10", "Arah Teori": "Protektif"},
            {"No": 9, "Instrumen": "KDRT", "Subskala": "Fisik - Favorable", "Jumlah Item": 4, "Skala": "1-4", "Arah Teori": "Risiko"},
            {"No": 10, "Instrumen": "KDRT", "Subskala": "Fisik - Unfavorable", "Jumlah Item": 6, "Skala": "1-4", "Arah Teori": "Protektif"},
            {"No": 11, "Instrumen": "KDRT", "Subskala": "Seksual - Favorable", "Jumlah Item": 3, "Skala": "1-4", "Arah Teori": "Risiko"},
            {"No": 12, "Instrumen": "KDRT", "Subskala": "Seksual - Unfavorable", "Jumlah Item": 6, "Skala": "1-4", "Arah Teori": "Protektif"},
            {"No": 13, "Instrumen": "KDRT", "Subskala": "Memata-matai - Favorable", "Jumlah Item": 3, "Skala": "1-4", "Arah Teori": "Risiko"},
            {"No": 14, "Instrumen": "KDRT", "Subskala": "Memata-matai - Unfavorable", "Jumlah Item": 5, "Skala": "1-4", "Arah Teori": "Protektif"},
            {"No": 15, "Instrumen": "KDRT", "Subskala": "Psikologis - Favorable", "Jumlah Item": 6, "Skala": "1-4", "Arah Teori": "Risiko"},
            {"No": 16, "Instrumen": "KDRT", "Subskala": "Psikologis - Unfavorable", "Jumlah Item": 7, "Skala": "1-4", "Arah Teori": "Protektif"},
        ])
        st.dataframe(struktur, hide_index=True, use_container_width=True)

        st.markdown("""
        **Catatan istilah**:
        - Favorable (pro-kekerasan): item yang jika disetujui berarti mendukung kekerasan. Arah teori adalah risiko.
        - Unfavorable (anti-kekerasan): item yang jika disetujui berarti menolak kekerasan. Arah teori adalah protektif.
        """)

    with sub4:
        st.subheader("Hasil Regresi Logistik 16 Fitur")
        st.caption("Koefisien standardized dengan 95 persen Bootstrap CI (1000 iterasi).")

        hasil_show = T2_HASIL.sort_values("skor_dominan", ascending=False).reset_index(drop=True)
        st.dataframe(
            hasil_show[["Fitur", "kategori", "beta", "ci_low", "ci_high", "importance", "skor_dominan"]]
                .style.format({"beta": "{:+.4f}", "ci_low": "{:+.4f}",
                               "ci_high": "{:+.4f}", "importance": "{:+.4f}",
                               "skor_dominan": "{:.4f}"}),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Coefficient Plot dengan Bootstrap CI")

        plot_df = T2_HASIL.sort_values("beta").copy()
        colors = plot_df["kategori"].map({
            "RISIKO": "#ef4444", "PROTEKTIF": "#10b981", "NETRAL": "#9ca3af",
        })
        fig, ax = plt.subplots(figsize=(11, 7))
        y_pos = np.arange(len(plot_df))
        ax.barh(y_pos, plot_df["beta"], color=colors, edgecolor="black", alpha=0.85)
        ax.errorbar(plot_df["beta"], y_pos,
                    xerr=[plot_df["beta"] - plot_df["ci_low"],
                          plot_df["ci_high"] - plot_df["beta"]],
                    fmt="none", ecolor="black", capsize=4, lw=1.2)
        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["Fitur"], fontsize=9)
        ax.set_xlabel("Koefisien Logistik (standardized)")
        ax.set_title("Faktor Risiko dan Protektif terhadap Perceraian (95% Bootstrap CI)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **Cara membaca**:
        - Bar hijau (PROTEKTIF): koefisien negatif, CI tidak melintasi nol. Faktor protektif.
        - Bar merah (RISIKO): koefisien positif, CI tidak melintasi nol. Faktor risiko.
        - Bar abu-abu (NETRAL): CI melintasi nol. Arah tidak dapat dipastikan.
        """)

    with sub5:
        st.subheader("Interpretasi Klasifikasi Tiga Tingkat")

        st.markdown("""
        **Aturan klasifikasi**:
        - RISIKO: koefisien positif dan interval kepercayaan tidak melintasi nol.
        - PROTEKTIF: koefisien negatif dan interval kepercayaan tidak melintasi nol.
        - NETRAL: interval kepercayaan melintasi nol.

        **Mengapa tiga tingkat, bukan dua**:
        Data sampel kecil (43 kasus cerai) menyebabkan banyak interval kepercayaan melebar. Jika dipaksa dua tingkat (signifikan atau tidak), banyak faktor yang sebenarnya menarik jadi terbuang. Klasifikasi netral memberi ruang untuk melaporkan arah kecenderungan tanpa overclaim.
        """)

        st.divider()
        st.markdown("### Faktor Risiko (signifikan)")
        risk_df = T2_HASIL[T2_HASIL["kategori"] == "RISIKO"].sort_values("skor_dominan", ascending=False)
        for _, r in risk_df.iterrows():
            with st.expander(f"{r['Fitur']} - beta = {r['beta']:+.3f}"):
                st.markdown(render_narasi_t2(r))
                st.markdown(f"- 95% CI: [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
                st.markdown(f"- Skor dominansi: {r['skor_dominan']:.3f}")

        st.markdown("### Faktor Protektif (signifikan)")
        prot_df = T2_HASIL[T2_HASIL["kategori"] == "PROTEKTIF"].sort_values("skor_dominan", ascending=False)
        for _, r in prot_df.iterrows():
            with st.expander(f"{r['Fitur']} - beta = {r['beta']:+.3f}"):
                st.markdown(render_narasi_t2(r))
                st.markdown(f"- 95% CI: [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
                st.markdown(f"- Skor dominansi: {r['skor_dominan']:.3f}")

        st.markdown("### Faktor Netral (CI melintasi nol)")
        netral_df = T2_HASIL[T2_HASIL["kategori"] == "NETRAL"]
        for _, r in netral_df.iterrows():
            with st.expander(f"{r['Fitur']} - beta = {r['beta']:+.3f}"):
                lean = classify_lean(r["beta"], r["ci_low"], r["ci_high"])
                st.markdown(f"**Kecenderungan**: {lean}")
                st.markdown(f"- 95% CI: [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
                st.markdown(render_narasi_t2(r))

        st.divider()
        st.markdown("""
        **Mengapa banyak faktor netral condong ke aman**:

        Beberapa faktor seperti KDRT Fisik Favorable, KDRT Memata-matai Favorable, dan KDRT Seksual Favorable menunjukkan koefisien negatif (condong protektif), namun interval kepercayaannya sangat lebar sehingga tidak signifikan. Ini bisa terjadi karena:

        1. Sampel kecil (43 kasus cerai) membuat interval kepercayaan melebar.
        2. Multikolinearitas antar subskala KDRT menyebabkan efek dibagi antar subskala.
        3. Kemungkinan efek sebenarnya ada namun tidak terdeteksi pada sampel ini.

        Faktor-faktor ini tidak bisa diklaim sebagai protektif signifikan, namun juga tidak bisa diklaim sebagai tidak penting. Kondisi ini dilaporkan sebagai keterbatasan penelitian.
        """)

    with sub6:
        st.subheader("Stabilitas Koefisien di Repeated 5-fold x 10 CV")
        st.markdown("""
        Stabilitas diuji dengan mengulang cross-validation 10 kali dengan 5 lipatan setiap ulangan, menghasilkan 50 kali evaluasi. Proporsi arah sama menunjukkan seberapa konsisten tanda koefisien di berbagai split data.
        """)
        st.dataframe(
            T2_STABILITAS.style.format({"beta_full": "{:+.3f}", "beta_cv": "{:+.3f}",
                                         "sd": "{:.3f}", "proporsi": "{:.3f}"}),
            use_container_width=True,
        )

        st.markdown("""
        **Interpretasi**:
        - Proporsi di atas 0.9: koefisien stabil, arah konsisten di berbagai split data.
        - Proporsi 0.8 sampai 0.9: stabil marginal.
        - Proporsi di bawah 0.8: tidak stabil, jangan diinterpretasi.

        Faktor dengan proporsi terendah adalah KDRT Memata-matai Favorable (0.64), yang berarti koefisiennya fluktuatif dan tidak dapat disimpulkan.
        """)

    with sub7:
        st.subheader("Korelasi Antar 16 Fitur")
        st.markdown("""
        4 pasang fitur dengan korelasi di atas 0.7:
        - Shared Meaning dengan Love Maps: r = 0.757
        - KDRT Fisik Unfavorable dengan KDRT Seksual Unfavorable: r = 0.720
        - KDRT Fisik Unfavorable dengan KDRT Psikologis Unfavorable: r = 0.731
        - KDRT Seksual Unfavorable dengan KDRT Psikologis Unfavorable: r = 0.732

        Korelasi tinggi tidak di-drop karena tiap subskala mengukur dimensi berbeda. Namun hal ini menjadi keterbatasan penelitian yang perlu disebutkan.
        """)

        st.subheader("VIF Setiap Fitur")
        st.dataframe(T2_VIF, hide_index=True, use_container_width=True)


# ==========================================================
# TUJUAN 3 - ANALISIS INTERAKSI
# ==========================================================
with main_tab3:
    st.header("Tujuan 3: Analisis Interaksi (Moderasi)")
    st.caption("6 hipotesis interaksi dengan regresi logistik. Koreksi FDR untuk multiple testing.")

    sub1, sub2, sub3, sub4, sub5, sub6 = st.tabs([
        "Metodologi",
        "Uji Asumsi Awal",
        "6 Hipotesis",
        "Hasil Interaksi",
        "Simple Slopes",
        "Stabilitas CV",
    ])

    with sub1:
        st.subheader("Metodologi Tujuan 3")
        st.markdown("""
        **Tujuan**: menguji apakah hubungan antara faktor risiko/protektif dengan perceraian berbeda menurut tingkat moderator (sosial-ekonomi, keuangan, sikap KDRT).

        **Pendekatan**:
        1. Model regresi logistik dengan interaction term: logit(Y) = beta_0 + beta_1 X1 + beta_2 X2 + beta_3 (X1 x X2).
        2. X1 adalah prediktor utama (Shared Meaning, Negative Conflict, Defensiveness).
        3. X2 adalah moderator (Pendapatan, Pendidikan, Financial Stress, KDRT Permisif).
        4. Koefisien beta_3 adalah koefisien interaksi yang diuji.
        5. Signifikansi diuji dengan Likelihood Ratio Test (membandingkan model dengan dan tanpa interaksi) dan Wald test.
        6. Koreksi FDR Benjamini-Hochberg diterapkan karena 6 hipotesis diuji bersamaan.
        7. Semua prediktor dan moderator distandardisasi untuk menyeragamkan skala.

        **Interpretasi arah koefisien interaksi**:
        - Koefisien positif: efek X1 menguat saat X2 meningkat.
        - Koefisien negatif: efek X1 melemah saat X2 meningkat.

        **Mengapa koreksi FDR**:
        Menguji 6 hipotesis secara bersamaan meningkatkan risiko false positive. Koreksi FDR mengontrol proporsi false discovery. Tanpa koreksi, satu hasil signifikan dari enam hipotesis bisa jadi kebetulan statistik.
        """)

    with sub2:
        st.subheader("Uji Asumsi Awal Data")
        asumsi_t3 = pd.DataFrame([
            {"No": 1, "Uji": "Missing values", "Hasil": "0 nilai hilang", "Status": "Baik",
             "Penjelasan": "Data lengkap"},
            {"No": 2, "Uji": "Distribusi target", "Hasil": "460 menikah vs 43 cerai", "Status": "Tidak seimbang",
             "Penjelasan": "Imbalance 10.7 banding 1, ditangani dengan class_weight balanced"},
            {"No": 3, "Uji": "Outlier (IQR)", "Hasil": "0 sampai 9.94 persen per fitur", "Status": "Wajar",
             "Penjelasan": "Outlier pada skala Likert tidak di-drop karena bisa merepresentasikan kasus valid"},
            {"No": 4, "Uji": "Korelasi antar fitur", "Hasil": "Semua di bawah 0.7", "Status": "Baik",
             "Penjelasan": "Tidak ada korelasi tinggi antar prediktor dan moderator"},
            {"No": 5, "Uji": "Kolinearitas sempurna", "Hasil": "Financial Well-Being vs Financial Stress r = -1.000", "Status": "Perlu penanganan",
             "Penjelasan": "Financial Well-Being di-drop, hanya Financial Stress yang dipakai"},
        ])
        st.dataframe(asumsi_t3, hide_index=True, use_container_width=True)

        st.markdown("""
        
        **Mengapa outlier tidak di-drop**:
        Pada data survei dengan skala Likert, nilai ekstrem dapat merepresentasikan kasus yang valid secara substantif, bukan kesalahan input. Menghapusnya tanpa alasan kuat bisa bias.

        **Setelah handling**:
        VIF fitur utama maksimal 2.18 (aman). VIF interaksi maksimal 1.31 (aman). EPV = 10.75 (borderline, ideal 10).
        """)

    with sub3:
        st.subheader("6 Hipotesis Interaksi yang Diuji")
        hipotesis_df = pd.DataFrame([
            {"No": 1, "Kelompok": "Sosial ekonomi", "Interaksi": "Shared Meaning x Pendapatan",
             "Pertanyaan": "Apakah hubungan Shared Meaning dengan cerai berbeda menurut pendapatan?"},
            {"No": 2, "Kelompok": "Sosial ekonomi", "Interaksi": "Negative Conflict x Pendidikan",
             "Pertanyaan": "Apakah hubungan konflik negatif dengan cerai berbeda menurut pendidikan?"},
            {"No": 3, "Kelompok": "Keuangan", "Interaksi": "Negative Conflict x Financial Stress",
             "Pertanyaan": "Apakah hubungan konflik negatif dengan cerai berbeda menurut stres keuangan?"},
            {"No": 4, "Kelompok": "Keuangan", "Interaksi": "Shared Meaning x Financial Stress",
             "Pertanyaan": "Apakah hubungan shared meaning dengan cerai berbeda menurut stres keuangan?"},
            {"No": 5, "Kelompok": "Sikap KDRT", "Interaksi": "Negative Conflict x KDRT Permisif",
             "Pertanyaan": "Apakah hubungan konflik negatif dengan cerai berbeda menurut sikap permisif KDRT?"},
            {"No": 6, "Kelompok": "Sikap KDRT", "Interaksi": "Defensiveness x KDRT Permisif",
             "Pertanyaan": "Apakah hubungan defensiveness dengan cerai berbeda menurut sikap permisif KDRT?"},
        ])
        st.dataframe(hipotesis_df, hide_index=True, use_container_width=True)

    with sub4:
        st.subheader("Hasil Uji 6 Hipotesis Interaksi")
        st.dataframe(
            T3_HASIL.style.format({"beta": "{:+.4f}", "ci_low": "{:+.4f}",
                                    "ci_high": "{:+.4f}", "p_Wald": "{:.4f}",
                                    "p_LRT": "{:.4f}", "p_FDR": "{:.4f}"}),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Forest Plot Koefisien Interaksi")

        plot_df = T3_HASIL.sort_values("beta").copy()
        colors = plot_df["Status"].map({
            "Signifikan Raw": "#ef4444", "Tidak Signifikan": "#9ca3af"})
        fig, ax = plt.subplots(figsize=(11, 5))
        y_pos = np.arange(len(plot_df))
        ax.barh(y_pos, plot_df["beta"], color=colors, edgecolor="black", alpha=0.85)
        ax.errorbar(plot_df["beta"], y_pos,
                    xerr=[plot_df["beta"] - plot_df["ci_low"],
                          plot_df["ci_high"] - plot_df["beta"]],
                    fmt="none", ecolor="black", capsize=4, lw=1.2)
        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["Hipotesis"], fontsize=10)
        ax.set_xlabel("Koefisien Interaksi (standardized)")
        ax.set_title("Forest Plot 6 Hipotesis Interaksi")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.markdown("### Ringkasan Signifikansi")
        c1, c2, c3 = st.columns(3)
        c1.metric("Signifikan raw (p < 0.05)", "1 dari 6")
        c2.metric("Signifikan setelah FDR", "0 dari 6")
        c3.metric("Signifikan setelah Bonferroni", "0 dari 6")

        st.markdown("""
        **Narasi tiap hipotesis**:
        - Shared x Pendapatan: beta = -0.243, p = 0.266. Tidak signifikan. Tidak ada bukti hubungan shared meaning dengan cerai berbeda menurut pendapatan.
        - Konflik x Pendidikan: beta = -0.087, p = 0.572. Tidak signifikan. Tidak ada bukti hubungan konflik dengan cerai berbeda menurut pendidikan.
        - Konflik x Financial Stress: beta = -0.214, p = 0.136. Tidak signifikan. Tidak ada bukti interaksi.
        - Shared x Financial Stress: beta = +0.341, p (raw) = 0.026, p (FDR) = 0.155. Signifikan secara raw, tetapi tidak lolos koreksi FDR.
        - Konflik x KDRT Permisif: beta = -0.016, p = 0.923. Tidak signifikan.
        - Defensif x KDRT Permisif: beta = -0.125, p = 0.405. Tidak signifikan.
        """)

    with sub5:
        st.subheader("Simple Slopes: Shared Meaning x Financial Stress")
        st.caption("Analisis eksploratif. Hipotesis ini signifikan raw tetapi tidak lolos koreksi FDR.")

        st.markdown("""
        **Catatan arah skala IFDFW**:

        Skala asli IFDFW adalah 1 sampai 10, di mana nilai 10 berarti kondisi
        keuangan sangat baik (tidak stres) dan nilai 1 berarti kondisi
        keuangan sangat buruk (sangat stres).

        Dalam analisis, variabel Financial Stress dihitung sebagai
        `11 - rata-rata IFDFW`, sehingga:

        - Financial Stress rendah = kondisi keuangan BAIK (nilai mentah tinggi)
        - Financial Stress tinggi = kondisi keuangan BURUK (nilai mentah rendah)

        Interpretasi simple slopes di bawah mengikuti arah variabel
        Financial Stress yang sudah direverse ini.
        """)

        simple_df = pd.DataFrame([
            {"Level Financial Stress": "Rendah (mean - 1 SD)",
             "Kondisi Keuangan": "Baik (nilai mentah IFDFW tinggi)",
             "Slope Shared Meaning": -1.600,
             "CI_Low": -2.119, "CI_High": -1.081, "p_value": "< 0.001"},
            {"Level Financial Stress": "Sedang (mean)",
             "Kondisi Keuangan": "Sedang (nilai mentah IFDFW sedang)",
             "Slope Shared Meaning": -1.259,
             "CI_Low": -1.589, "CI_High": -0.929, "p_value": "< 0.001"},
            {"Level Financial Stress": "Tinggi (mean + 1 SD)",
             "Kondisi Keuangan": "Buruk (nilai mentah IFDFW rendah)",
             "Slope Shared Meaning": -0.919,
             "CI_Low": -1.296, "CI_High": -0.541, "p_value": "< 0.001"},
        ])
        st.dataframe(simple_df, hide_index=True, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(3)
        ax.plot(x, simple_df["Slope Shared Meaning"], "o-", color="#3b82f6", lw=2, markersize=10)
        ax.fill_between(x, simple_df["CI_Low"], simple_df["CI_High"], alpha=0.2, color="#3b82f6")
        ax.set_xticks(x)
        ax.set_xticklabels(["Kondisi\nBaik", "Kondisi\nSedang", "Kondisi\nBuruk"])
        ax.axhline(0, color="black", ls="--", alpha=0.5)
        ax.set_ylabel("Slope Shared Meaning")
        ax.set_xlabel("Kondisi Keuangan (dari variabel Financial Stress)")
        ax.set_title("Simple Slopes: Shared Meaning vs Financial Stress")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **Narasi simple slopes**:

        - Pada kondisi keuangan baik (Financial Stress rendah, −1 SD):
          slope shared meaning = −1.600. Setiap kenaikan 1 SD shared meaning
          menurunkan log-odds cerai sekitar 1.60. Efek protektifnya paling
          kuat pada kondisi ini.

        - Pada kondisi keuangan sedang (Financial Stress mean):
          slope shared meaning = −1.259.

        - Pada kondisi keuangan buruk (Financial Stress tinggi, +1 SD):
          slope shared meaning = −0.919. Kekuatan protektifnya melemah,
          namun masih tetap signifikan.

        **Kesimpulan**: shared meaning tetap protektif di semua level
        kondisi keuangan. Namun kekuatan protektifnya berkurang ketika
        kondisi keuangan memburuk. Dengan kata lain, kesulitan keuangan
        melemahkan efektivitas shared meaning sebagai faktor protektif.

        **Catatan penting**: temuan ini bersifat eksploratif karena tidak
        lolos koreksi FDR (p_FDR = 0.155). Perlu replikasi pada sampel
        independen sebelum dianggap sebagai temuan konfirmatori.
        """)

    with sub6:
        st.subheader("Stabilitas Koefisien Interaksi (Repeated 5-fold x 10 CV)")
        st.dataframe(
            T3_STABILITAS.style.format({"beta_full": "{:+.3f}", "beta_cv": "{:+.3f}",
                                         "sd": "{:.3f}", "proporsi": "{:.2f}"}),
            use_container_width=True,
        )

        st.markdown("""
        **Interpretasi**:
        - Shared x Stress: proporsi arah sama = 1.00. Sangat stabil, arah positif konsisten di seluruh 50 split data.
        - Konflik x Stress: proporsi = 1.00. Sangat stabil.
        - Shared x Pendapatan, Defensif x Permisif: proporsi di atas 0.95. Stabil.
        - Konflik x Permisif: proporsi = 0.54. Tidak stabil, koefisien fluktuatif dan tanda sering berubah. Jangan diinterpretasi meski angkanya terlihat ada.
        """)

        st.divider()
        st.markdown("""
        **Kesimpulan Tujuan 3**:

        1. Tidak ada interaksi yang signifikan setelah koreksi FDR untuk 6 hipotesis.
        2. Hanya Shared Meaning x Financial Stress yang signifikan secara raw (beta = +0.341, p = 0.026), namun setelah koreksi FDR menjadi p = 0.155. Ini menunjukkan temuan tersebut kemungkinan besar adalah false positive akibat multiple testing.
        3. Simple slopes menunjukkan pola yang menarik: efek protektif shared meaning melemah saat stres keuangan meningkat. Namun, temuan ini bersifat eksploratif.
        4. Sebagian besar moderator (pendapatan, pendidikan, KDRT permisif) tidak memoderasi hubungan antara faktor risiko/protektif dengan perceraian.

        **Keterbatasan utama**: sampel cerai hanya 43 orang. Uji interaksi memerlukan sampel 4 kali lebih besar dibanding uji efek utama untuk mencapai power 0.80. Penelitian ini hanya memiliki power sekitar 30 sampai 40 persen untuk mendeteksi interaksi berukuran sedang.
        """)


# ==========================================================
# FOOTER
# ==========================================================
st.divider()
st.caption("Dashboard Penelitian. Hasil bersifat indikatif untuk keperluan penelitian akademik.")
