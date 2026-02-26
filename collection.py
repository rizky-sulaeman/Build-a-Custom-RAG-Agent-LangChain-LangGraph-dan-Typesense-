import os, json, math, requests, typesense  

api_key = os.getenv("TYPESENSE_API_KEY")  # ambil API key dari environment variable

client = typesense.Client({
    # konfigurasi koneksi ke Typesense lokal
    "nodes": [{"host": "localhost", "port": "8108", "protocol": "http"}],
    "api_key": api_key,
    "connection_timeout_seconds": 10,
})

""
def delete_collection_if_exists(name: str):
    try:
        client.collections[name].delete()
        print(f"Collection '{name}' dihapus (jika ada).")
    except Exception:
        print(f"Collection '{name}' belum ada, akan dibuat.")


def setup_faqs_collection():
    print("faqs \n")  # info proses
    delete_collection_if_exists("faqs")  # hapus collection lama kalau perlu

    schema = {
        "name": "faqs",
        "fields": [
            {"name": "id", "type": "string"},  # id unik untuk tiap dokumen
            {"name": "prompt", "type": "string"},  # pertanyaan
            {"name": "completion", "type": "string"},  # jawaban
        ],
    }
    client.collections.create(schema)  # bikin collection baru

    docs = []
    with open("faqs_extend_no_split.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip baris kosong
            d = json.loads(line)
            d["id"] = str(i)  # kasih id urut
            docs.append(d)

    res = client.collections["faqs"].documents.import_(docs, {"action": "create"})  # import ke Typesense
    ok = sum(1 for r in res if r.get("success"))  # hitung yang sukses
    fail = len(res) - ok
    print(f"   Import faqs: {ok} sukses, {fail} gagal, total {len(docs)}")
    if fail:
        for r in res:
            if not r.get("success"):
                print("Contoh error:", r)  # tampilkan error 
                break


def setup_hospitals_collection():
    print("hospitals \n")  # info proses
    delete_collection_if_exists("hospitals")  # hapus collection lama kalau perlu

    schema = {
        "name": "hospitals",
        "fields": [
            {"name": "id", "type": "string"},  # id unik untuk tiap dokumen
            {"name": "no", "type": "int32"},  # nomor urut
            {"name": "hospital_id", "type": "string"},  # id rumah sakit
            {"name": "hospital", "type": "string"},  # nama rumah sakit
            {"name": "hospital_2", "type": "string", "optional": True},  # nama alternatif rumah sakit
            {"name": "alias", "type": "string"},  # alias
            {"name": "address", "type": "string"},  # alamat
            {"name": "district", "type": "string"},  # kecamatan
            {"name": "city", "type": "string"},  # kota
            {"name": "province", "type": "string"},  # provinsi
            {"name": "slug", "type": "string", "optional": True},  # slug url
            {"name": "lng", "type": "float", "optional": True},  # longitude
            {"name": "lat", "type": "float", "optional": True},  # latitude
        ],
    }
    client.collections.create(schema)  # bikin collection baru

    with open("hospitals_prod.json", encoding="utf-8") as f:
        raw = json.load(f)  # load data json

    docs = []
    for item in raw:
        d = {
            "id": str(item["No"]),
            "no": item["No"],
            "hospital_id": item["Id"],
            "hospital": item["Hospital"],
            "alias": item.get("Alias", ""),
            "address": item.get("Address", ""),
            "district": item.get("District", ""),
            "city": item.get("City", ""),
            "province": item.get("Province", ""),
            "slug": item.get("Slug", ""),
            "lng": item.get("lng"),
            "lat": item.get("lat"),
        }
        if item.get("Hospital_2"):
            d["hospital_2"] = item["Hospital_2"]  # tambahkan jika ada
        docs.append(d)

    res = client.collections["hospitals"].documents.import_(docs, {"action": "create"})  # import ke Typesense
    ok = sum(1 for r in res if r.get("success")) # hitung yang sukses
    fail = len(res) - ok
    print(f"Import hospitals: {ok} sukses, {fail} gagal, total {len(docs)}")
    if fail:
        for r in res:
            if not r.get("success"):
                print("Contoh error:", r) # tampilkan error 
                break


def setup_doctors_collection():
    print("doctors \n")  # info proses
    delete_collection_if_exists("doctors")  # hapus collection lama kalau perlu

    schema = {
        "name": "doctors",
        "fields": [
            {"name": "id", "type": "string"},  # id urut unik untuk tiap dokumen
            {"name": "doctor_id", "type": "string"},  # id dokter
            {"name": "name", "type": "string"},  # nama dokter
            {"name": "doctor_hope_id", "type": "int64", "optional": True},  # id hope (opsional)
            {"name": "gender_name", "type": "string", "optional": True},  # jenis kelamin
            {"name": "specialization_id", "type": "string", "optional": True},  # id spesialisasi
            {"name": "specialization_name", "type": "string", "optional": True},  # nama spesialisasi
            {"name": "specialization_name_en", "type": "string", "optional": True},  # nama spesialisasi (en)
            {"name": "sub_specialization_name", "type": "string", "optional": True},  # subspesialisasi
            {"name": "sub_specialization_name_en", "type": "string", "optional": True},  # subspesialisasi (en)
            {"name": "image_url", "type": "string", "optional": True},  # foto
            {"name": "is_emergency_enable", "type": "bool", "optional": True},  # bisa emergency?
            {"name": "consultation_price", "type": "int64", "optional": True},  # harga konsultasi
            {"name": "teleconsult_price", "type": "int64", "optional": True},  # harga telekonsultasi
            {"name": "is_have_schedule", "type": "bool", "optional": True},  # punya jadwal?
            {"name": "consultation_type", "type": "string", "optional": True},  # tipe konsultasi
            {"name": "doctor_seo_key", "type": "string", "optional": True},  # seo key maksudnya adalah bagian dari url dokter yang unik, biasanya berupa nama yang sudah diubah jadi lowercase dan diganti spasi dengan tanda hubung, contoh: "dr-xyz-spesialis-kulit"
            {"name": "next_avail", "type": "string", "optional": True},  # jadwal berikutnya
            {"name": "hospital_names", "type": "string[]", "optional": True},  # list nama RS
            {"name": "hospital_aliases", "type": "string[]", "optional": True},  # list alias RS
        ],
    }
    client.collections.create(schema)  # bikin collection baru

    print("Mengambil data dokter dari file doctors.json...")
    with open("doctors.json", encoding="utf-8") as f:
        doctors_json = json.load(f)
    all_docs = doctors_json.get("data", []) if isinstance(doctors_json, dict) else doctors_json
    print(f"Total diambil: {len(all_docs)}")

    docs = []
    for i, doc in enumerate(all_docs, start=1):
        hosp_names, hosp_aliases = [], []
        for h in doc.get("hospital_ids") or []:
            if h.get("hospital_name"):
                hosp_names.append(h["hospital_name"])
            if h.get("alias"):
                hosp_aliases.append(h["alias"])

        t = {
            "id": str(i),
            "doctor_id": doc.get("doctor_id", ""),
            "name": doc.get("name", ""),
            "specialization_name": doc.get("specialization_name") or "",
            "specialization_name_en": doc.get("specialization_name_en") or "",
            "sub_specialization_name": doc.get("sub_specialization_name") or "",
            "sub_specialization_name_en": doc.get("sub_specialization_name_en") or "",
            "image_url": doc.get("image_url") or "",
            "is_emergency_enable": doc.get("is_emergency_enable", False),
            "consultation_price": doc.get("consultation_price") or 0,
            "teleconsult_price": doc.get("teleconsult_price") or 0,
            "is_have_schedule": doc.get("is_have_schedule", False),
            "consultation_type": doc.get("consultation_type") or "",
            "doctor_seo_key": doc.get("doctor_seo_key") or "",
            "next_avail": doc.get("next_avail") or "",
            "hospital_names": hosp_names,
            "hospital_aliases": hosp_aliases,
        }
        if doc.get("doctor_hope_id") is not None:
            t["doctor_hope_id"] = doc["doctor_hope_id"]  # tambahkan jika ada
        if doc.get("gender_name"):
            t["gender_name"] = doc["gender_name"]
        if doc.get("specialization_id"):
            t["specialization_id"] = doc["specialization_id"]
        docs.append(t)

    batch_size = 250  # biar ga terlalu besar sekali import
    total_ok = total_fail = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        res = client.collections["doctors"].documents.import_(batch, {"action": "create"})
        ok = sum(1 for r in res if r.get("success"))
        fail = len(res) - ok
        total_ok += ok
        total_fail += fail
        if fail:
            for r in res:
                if not r.get("success"):
                    print(f"Error batch {i // batch_size + 1}:", r)  # tampilkan error
                    break

    print(f"Import doctors: {total_ok} sukses, {total_fail} gagal, total {len(docs)}")


if __name__ == "__main__":
    # jalankan semua setup collection sekaligus
    setup_faqs_collection()
    setup_hospitals_collection()
    setup_doctors_collection()
