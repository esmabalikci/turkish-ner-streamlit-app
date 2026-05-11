# Turkish Named Entity Recognition (NER) Project

Bu proje, Doğal Dil İşleme dersi kapsamında geliştirilmiş Türkçe İsim Varlık Tanıma (NER) uygulamasıdır.

Proje, kullanıcı tarafından girilen Türkçe metin içerisindeki varlıkları tespit eder ve aşağıdaki kategorilere ayırır:

- KİŞİ
- YER
- KURUM

## Kullanılan Teknolojiler

- Python
- Streamlit
- Hugging Face Transformers
- BERT
- Datasets Library

## Kullanılan Model

Bu projede aşağıdaki ön eğitimli model kullanılmıştır:

savasy/bert-base-turkish-ner-cased

## Proje Dosyaları

- `app.py` → Streamlit kullanıcı arayüzü
- `train.py` → Model eğitim kodu
- `data.json` → Eğitim veri seti
## Kurulum

Gerekli kütüphaneleri yükleyin:

```bash
pip install streamlit transformers datasets torch
```

## Model Eğitimi

Model klasörünü oluşturmak için:

```bash
python train.py
```

Bu işlem sonunda `ner_model/` klasörü oluşacaktır.

## Uygulamayı Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda açılan ekrandan metin girerek analiz yapabilirsiniz.

## Örnek Kullanım

Girdi:

```text
Mehmet Ankara Hacettepe Üniversitesi
```

Çıktı:

```text
Mehmet → KİŞİ
Ankara → YER
Hacettepe Üniversitesi → KURUM
```

## Not

Model dosyaları yükleme boyutu nedeniyle projeye eklenmemiştir. Model yeniden `train.py` ile oluşturulabilir.