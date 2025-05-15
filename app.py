import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Konfigurasi direktori
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "static", "models", "mobilenetv2_finetune_model_daun.h5")

# Konfigurasi files
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok = True)

# Load model
model = load_model(MODEL_PATH)

# Label kelas
class_labels = [
	"Daun Jambu Biji", "Daun Kari", "Daun Kemangi", "Daun Kunyit", "Daun Mint",
	"Daun Pepaya", "Daun Sirih", "Daun Sirsak", "Lidah Buaya", "Teh Hijau"
]

# Daftar manfaat berdasarkan label
manfaat_dict = {
	"Daun Jambu Biji":	"Daun jambu biji telah lama digunakan dalam pengobatan tradisional sebagai antimikroba, antiinflamasi, dan antioksidan. Kandungan senyawa aktif seperti flavonoid (quercetin), tanin, dan saponin menjadikan daun ini sangat efektif dalam mengatasi gangguan pencernaan. Flavonoid memiliki efek antibakteri yang terbukti mampu melawan bakteri seperti Escherichia coli dan Salmonella—dua bakteri umum penyebab diare. Oleh karena itu, rebusan daun jambu biji sering dijadikan ramuan alami untuk mengobati diare dan disentri, terutama di daerah pedesaan.\n\n Manfaat daun jambu biji juga mencakup pengendalian kadar gula darah pada penderita diabetes tipe 2. Studi menunjukkan bahwa senyawa di dalamnya dapat meningkatkan sensitivitas insulin. Selain itu, daun ini memiliki potensi menurunkan kadar kolesterol LDL (jahat) dan meningkatkan kolesterol HDL (baik), yang membantu mencegah aterosklerosis. Tak hanya itu, senyawa antioksidannya membantu menangkal radikal bebas yang berperan dalam penuaan sel dan pembentukan sel kanker. Daun ini juga sering dipakai dalam produk perawatan kulit untuk mengatasi jerawat dan iritasi.",
	"Daun Kari":		"Daun kari bukan hanya bumbu dapur, tetapi juga tanaman obat yang sangat kaya manfaat. Dalam Ayurveda, daun kari digunakan sebagai tonik penambah nafsu makan, peluruh keringat, dan perangsang sistem pencernaan. Kandungan senyawa bioaktif seperti carbazole alkaloid (mahanimbine, koenimbine) menunjukkan aktivitas antibakteri, antijamur, antioksidan, dan antikanker. Daun ini diyakini membantu menurunkan kadar kolesterol serta melindungi jantung dari penyakit kardiovaskular.\n\n Penelitian juga membuktikan daun kari memiliki efek hipoglikemik yang baik dalam membantu mengontrol kadar gula darah. Carbazole alkaloid bekerja dengan meningkatkan sekresi insulin dan memperbaiki sensitivitasnya. Di samping itu, kandungan vitamin A dalam daun kari sangat baik untuk kesehatan mata, sedangkan zat besi dan kalsium mendukung fungsi tulang dan darah. Daun ini juga digunakan secara eksternal untuk mengobati luka, gigitan serangga, dan infeksi kulit ringan.",
	"Daun Kemangi":		"Kemangi merupakan anggota keluarga mint (Lamiaceae) yang kaya akan minyak esensial seperti eugenol, linalool, dan methyl chavicol. Eugenol memiliki aktivitas antimikroba yang sangat tinggi, bekerja melawan bakteri penyebab infeksi mulut, tenggorokan, dan saluran pencernaan. Dalam pengobatan tradisional, kemangi sering dikonsumsi untuk meredakan perut kembung, menenangkan otot saluran cerna, dan memperlancar pencernaan. Selain itu, daunnya digunakan sebagai ramuan untuk menyegarkan napas dan mengurangi bau badan secara alami. \n\n Kemangi juga memiliki efek adaptogenik, yaitu membantu tubuh beradaptasi terhadap stres dan meningkatkan keseimbangan hormon. Oleh karena itu, konsumsi kemangi sangat baik untuk menjaga suasana hati dan meningkatkan daya tahan tubuh. Dalam studi ilmiah, daun kemangi menunjukkan efek pelindung terhadap kerusakan hati dan ginjal akibat paparan racun, serta dapat menurunkan kadar gula darah dan kolesterol dalam darah. Kandungan antioksidan yang tinggi juga memberikan manfaat anti-aging bagi kulit dan sel tubuh.",
	"Daun Kunyit":		"Meskipun yang umum digunakan adalah rimpangnya, daun kunyit juga menyimpan manfaat luar biasa. Daun ini mengandung turunan kurkuminoid dan minyak atsiri yang memiliki efek antiinflamasi, hepatoprotektif, dan analgesik. Dalam pengobatan tradisional, daun kunyit sering digunakan untuk mengobati gangguan menstruasi, mengurangi rasa sakit, dan memperlancar peredaran darah. Rebusan daun kunyit kerap dijadikan minuman kesehatan untuk memperbaiki fungsi hati, detoksifikasi tubuh, dan menjaga kesehatan pencernaan. \n\n Selain itu, daun kunyit memiliki manfaat sebagai agen pelindung terhadap kanker. Penelitian menunjukkan bahwa ekstrak dari daun kunyit dapat menghambat proliferasi sel tumor serta menstimulasi apoptosis (kematian sel kanker). Daunnya juga digunakan dalam baluran atau ramuan luar untuk mempercepat penyembuhan luka, meredakan gatal, atau mengatasi infeksi kulit. Dalam jamu tradisional, daun ini kerap dicampur dengan bahan lain seperti daun sirih atau asam jawa untuk pengobatan wanita pasca melahirkan.",
	"Daun Mint":		"Mint dikenal sebagai herbal penenang dan pereda gangguan pencernaan sejak zaman kuno. Daun mint mengandung senyawa utama mentol, yang memberi efek dingin, menyegarkan, dan membantu mengendurkan otot polos di saluran pencernaan. Efek ini membuatnya sangat efektif untuk mengatasi kembung, mual, kolik, dan irritable bowel syndrome (IBS). Mint juga membantu meningkatkan produksi empedu, yang mendukung proses pencernaan lemak. \n\n Tak hanya bermanfaat untuk sistem pencernaan, daun mint juga populer sebagai herbal untuk pernapasan. Uap mint dapat melegakan hidung tersumbat, bronkitis, dan radang tenggorokan. Dalam bidang neurologi, mint telah terbukti membantu mengurangi sakit kepala tegang dan migrain ketika dioleskan dalam bentuk minyak. Kandungan antioksidan dan vitamin C pada daun mint membantu memperkuat daya tahan tubuh serta menjaga kesegaran kulit dan napas.",
	"Daun Pepaya":		"Daun pepaya mengandung enzim papain dan chymopapain yang berfungsi sebagai protease, membantu pemecahan protein dalam sistem pencernaan. Daun ini dikenal luas sebagai obat alami penambah trombosit, terutama dalam kasus demam berdarah dengue (DBD). Kandungan acetogenin dan flavonoidnya bekerja untuk meningkatkan produksi trombosit dan melindungi sumsum tulang dari kerusakan oleh virus.\n\n Selain itu, daun pepaya memiliki aktivitas antiparasit yang membantu mengeliminasi cacing dalam saluran pencernaan. Dengan kandungan antioksidan tinggi seperti vitamin C, E, dan beta-karoten, daun ini juga melindungi sel tubuh dari kerusakan oksidatif dan membantu proses penyembuhan luka. Dalam pengobatan tradisional, jus daun pepaya dikonsumsi sebagai tonik herbal yang mendetoksifikasi hati dan memperbaiki kualitas darah.",
	"Daun Sirih":		"Daun sirih sudah digunakan sejak ratusan tahun oleh masyarakat Asia Tenggara sebagai antiseptik alami. Kandungan eugenol, chavicol, dan estragol memberikan aktivitas antimikroba yang kuat terhadap bakteri, jamur, dan virus. Daun ini sering direbus dan digunakan untuk mencuci luka, meredakan iritasi kulit, atau dijadikan obat kumur untuk mengatasi sariawan, gusi berdarah, dan bau mulut.\n\n Dalam dunia kewanitaan, daun sirih sangat populer sebagai obat keputihan dan perawatan organ intim. Air rebusannya digunakan sebagai pembersih alami yang membantu menjaga keseimbangan flora mikroba di area genital. Kandungan antiinflamasi dalam daun sirih juga membantu mengatasi peradangan ringan seperti radang tenggorokan, batuk, dan flu. Bahkan, dalam pengobatan tradisional, daun sirih juga dipercaya mampu membantu mengatasi gangguan pernapasan dan rematik.",
	"Daun Sirsak":		"Daun sirsak banyak diteliti karena mengandung acetogenins, senyawa yang terbukti secara ilmiah memiliki aktivitas antitumor. Senyawa ini bekerja dengan menghambat enzim yang dibutuhkan oleh sel kanker untuk berkembang. Karena itu, ekstrak daun sirsak sering dijadikan suplemen alternatif bagi pasien kanker, terutama kanker payudara, prostat, dan pankreas. Meski demikian, penggunaannya tetap perlu dalam pengawasan karena efek samping potensial.\n\n Selain antikanker, daun sirsak juga memiliki efek penenang yang dapat membantu mengatasi insomnia, stres, dan gangguan saraf. Kandungan antiinflamasi dan analgesiknya efektif meredakan nyeri sendi, rematik, serta radang otot. Rebusan daun sirsak juga digunakan untuk menurunkan tekanan darah tinggi, mengontrol gula darah, dan mendukung sistem imun secara menyeluruh. Dalam pengobatan rakyat, daun ini disebut sebagai “daun sejuta manfaat.",
	"Lidah Buaya":		"Lidah buaya telah diakui sebagai tanaman ajaib yang bermanfaat untuk kulit, rambut, dan kesehatan internal. Gel dalam daunnya kaya akan polisakarida seperti acemannan yang membantu mempercepat penyembuhan luka bakar, luka terbuka, dan iritasi kulit. Selain itu, sifat antibakteri dan antijamurnya juga membuat lidah buaya efektif dalam mengatasi jerawat dan infeksi kulit ringan.\n\n Di bidang internal, jus lidah buaya digunakan untuk menenangkan saluran pencernaan, mengatasi konstipasi, dan memperbaiki penyerapan nutrisi. Senyawa antioksidan dan vitamin di dalamnya membantu membersihkan hati dan memperkuat sistem kekebalan tubuh. Lidah buaya juga meningkatkan kesehatan rambut dengan menutrisi folikel dan menjaga kelembapan kulit kepala. Karena itulah tanaman ini sangat populer dalam industri kosmetik, farmasi, hingga makanan sehat.",
	"Teh Hijau":		"Teh hijau telah lama digunakan dalam pengobatan Tiongkok dan Jepang sebagai minuman penyembuh. Kandungan katekin—terutama EGCG (Epigallocatechin Gallate)—adalah senyawa aktif utama yang berfungsi sebagai antioksidan kuat. EGCG telah terbukti secara ilmiah membantu menghambat pertumbuhan sel kanker, menurunkan kolesterol, serta mencegah pembentukan plak arteri yang dapat menyebabkan penyakit jantung.\n\n Selain itu, teh hijau meningkatkan metabolisme tubuh, mendukung pembakaran lemak, dan membantu mengontrol berat badan. Efek stimulannya pada sistem saraf membuatnya berguna untuk meningkatkan kewaspadaan dan daya konsentrasi, tanpa menyebabkan ketegangan seperti kopi. Teh hijau juga memiliki efek menenangkan berkat kandungan L-theanine yang membantu relaksasi otak. Dengan manfaatnya yang sangat luas, teh hijau menjadi bagian penting dari gaya hidup sehat masa kini."
}

# Fungsi Klasifikasi
def process_and_predict(img_path):
	img = image.load_img(img_path, target_size = (128, 128))
	img_array = image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis = 0) / 255
	prediction = model.predict(img_array)
	class_idx = np.argmax(prediction, axis = 1)[0]
	prob = round(prediction[0][class_idx] * 100, 2)
	return class_labels[class_idx], prob

# Fungsi Input Files
def allowed_file(filename):
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Routing
@app.route("/")
def home():
	return render_template("app.html")

@app.route("/classify")
def classify_page():
	return render_template("classify.html")

@app.route("/classify", methods = ["POST", "GET"])
def upload_file():
	if request.method == "GET":
		return render_template("classify.html")
	else:
		file = request.files["image"]
		if (file.filename == ""):
			return render_template("app.html", error = "Tidak ada file yang dipilih.")
		elif (not allowed_file(file.filename)):
			return render_template("app.html", error = "File tidak valid. Gunakan JPG atau PNG.")
		
		filename = secure_filename(file.filename).replace(" ", "_")
		image_path = os.path.join(UPLOAD_FOLDER, filename)
		file.save(image_path)

		label, prob = process_and_predict(image_path)
		manfaat = manfaat_dict.get(label, "Manfaat dari daun ini belum tersedia.")

		return render_template(
			"classify.html",
			image_file_name = filename,
			label = label,
			prob = prob,
			manfaat = manfaat
		)

@app.route("/classify/<filename>")
def send_file(filename):
	return send_from_directory(UPLOAD_FOLDER, filename)

if (__name__ == "__main__"):
	app.debug = True
	app.run(debug=True)