
import json
import os
from typing import List, Dict, Any


# Nama file FAQ, RS, dan dokter bisa diubah lewat environment variable
FAqs_FILE = os.getenv("FAQS_FILE", "faqs_extend_no_split.jsonl")
HOSPITALS_FILE = os.getenv("HOSPITALS_FILE", "hospitals_prod.json")
DOCTOR_FILE = os.getenv("DOCTOR_FILE", "doctors.json")

# Fungsi untuk membuat chunk dari data FAQ
def build_faq_chunks(limit: int | None = None) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []  # List penampung hasil chunk
    # if not os.path.exists(FAqs_FILE):  # Kalau file tidak ada, langsung return kosong
    #     return chunks
    with open(FAqs_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):  # Baca baris per baris
            if limit is not None and len(chunks) >= limit:  # Batasi jumlah chunk jika limit diberikan
                break
            line = line.strip()  # Bersihkan spasi
            if not line:
                continue  # Lewati baris kosong
            d = json.loads(line)  # Parse JSON
            prompt = d.get("prompt", "")  # Ambil pertanyaan
            completion = d.get("completion", "")  # Ambil jawaban
            content = f"Pertanyaan: {prompt}\nJawaban: {completion}"  # Format konten
            chunks.append(
                {
                    "id": f"{i}",  # ID unik per FAQ
                    "content": content,
                    "source": "faqs",
                    "faq_index": i,  # Index FAQ
                }
            )
    return chunks



# Fungsi untuk membuat chunk dari data rumah sakit
def build_hospital_chunks(limit: int | None = None) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []  # List penampung hasil chunk
    # if not os.path.exists(HOSPITALS_FILE):  # Kalau file tidak ada, return kosong
    #     return chunks

    with open(HOSPITALS_FILE, encoding="utf-8") as f:
        raw = json.load(f)  # Baca seluruh data JSON

    for item in raw:
        if limit is not None and len(chunks) >= limit:  # Batasi jumlah chunk jika limit diberikan
            break

        name = item.get("Hospital") or ""  # Nama RS
        alias = item.get("Alias") or ""  # Alias RS
        addr = item.get("Address") or ""  # Alamat RS
        district = item.get("District") or ""  # Kecamatan
        city = item.get("City") or ""  # Kota
        province = item.get("Province") or ""  # Provinsi

        parts = [
            f"Rumah Sakit: {name}",  # Awali dengan nama RS
        ]
        if alias:
            parts.append(f"Alias: {alias}")  # Tambahkan alias jika ada
        if addr:
            parts.append(f"Alamat: {addr}")  # Tambahkan alamat jika ada
        loc_parts = [p for p in [district, city, province] if p]  # Gabungkan lokasi
        if loc_parts:
            parts.append("Lokasi: " + ", ".join(loc_parts))

        content = ". ".join(parts)  # Gabungkan semua bagian jadi satu string

        chunks.append(
            {
                "id": f"{item.get('No')}",  # ID unik per RS
                "content": content,
                "source": "hospitals",
                "hospital_id": item.get("Id"),
                "no": item.get("No"),
                "city": city,
                "province": province,
            }
        )
    return chunks



# Fungsi untuk membuat chunk dari data dokter
def build_doctor_chunks(limit: int | None = None) -> List[Dict[str, Any]]:
    """
    Ambil data dokter dari file doctors.json, satu chunk per dokter
    """
    print("Load data dokter dari file doctors.json untuk build chunks...")  # Info proses
    import json
    with open(DOCTOR_FILE, encoding="utf-8") as f:
        doctors_json = json.load(f)  # Baca data dokter
    all_docs = doctors_json.get("data", []) if isinstance(doctors_json, dict) else doctors_json  # Handle format dict/list
    total = len(all_docs)
    effective_total = min(total, limit) if limit is not None else total  # Hitung jumlah yang akan diproses sesuai limit
    print(f"Total dokter: {total},fetch: {effective_total}")
    if limit is not None:
        all_docs = all_docs[:limit]  # Potong jika limit

    chunks: List[Dict[str, Any]] = []
    # if not os.path.exists(DOCTOR_FILE):  # Kalau file tidak ada, return kosong
    #     return chunks
 
    for i, doc in enumerate(all_docs, start=1):  # Loop tiap dokter
        if limit is not None and len(chunks) >= limit:
            break

        name = doc.get("name") or ""  # Nama dokter
        spec = doc.get("specialization_name") or ""  # Spesialisasi
        subspec = doc.get("sub_specialization_name") or ""  # Subspesialisasi
        gender = doc.get("gender_name") or ""  # Jenis kelamin

        hospitals = []  # List RS tempat praktek
        for h in doc.get("hospital_ids") or []:
            hospital_name = h.get("hospital_name") or ""
            alias = h.get("alias") or ""
            if hospital_name:
                if alias:
                    hospitals.append(f"{hospital_name} ({alias})")  # Format nama RS + alias
                else:
                    hospitals.append(hospital_name)

        parts = [f"Dokter: {name}"]  # Awali dengan nama dokter
        if gender:
            parts.append(f"Jenis kelamin: {gender}")
        if spec:
            parts.append(f"Spesialisasi: {spec}")
        if subspec:
            parts.append(f"Sub-spesialisasi: {subspec}")
        if hospitals:
            parts.append("Praktek di: " + ", ".join(hospitals))

        content = ". ".join(parts)  # Gabungkan semua bagian jadi satu string

        chunks.append(
            {
                "id": f"{i}",  # ID unik per dokter
                "content": content,
                "source": "doctors",
                "doctor_id": doc.get("doctor_id"),
                "specialization_name": spec,
                "sub_specialization_name": subspec,
                "hospital_names": hospitals,
            }
        )
    return chunks



# Fungsi untuk membuat semua chunk sekaligus (FAQ, RS, dokter)
def build_all_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []  # List penampung hasil akhir

    max_faq = 5  # Limit FAQ
    max_hosp = 5  # Limit RS
    max_doc = 5  # Limit dokter

    chunks.extend(build_faq_chunks(limit=max_faq))  # Tambahkan chunk FAQ
    chunks.extend(build_hospital_chunks(limit=max_hosp))  # Tambahkan chunk RS
    chunks.extend(build_doctor_chunks(limit=max_doc))  # Tambahkan chunk dokter
    return chunks



# Fungsi untuk menulis hasil chunk ke file JSONL
def write_chunks_jsonl(path: str = "chunks.jsonl") -> None:
    chunks = build_all_chunks()  # buat semua chunk
    print(f"Total chunks: {len(chunks)}")  # Info jumlah chunk
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")  # Tulis per baris
    print(f"Tersimpan ke {path}")  # Info selesai



# Kalau file ini dijalankan langsung, tulis chunk ke file
if __name__ == "__main__":
    write_chunks_jsonl()

