# Mount Drive first (same as above)
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/drive/MyDrive/consumercomplaints/raw
!wget -O /content/drive/MyDrive/consumercomplaints/complaints.csv.zip \
  https://files.consumerfinance.gov/ccdb/complaints.csv.zip
!unzip -o /content/drive/MyDrive/consumercomplaints/complaints.csv.zip -d /content/drive/MyDrive/consumercomplaints/raw
!ls -lh /content/drive/MyDrive/consumercomplaints/raw
