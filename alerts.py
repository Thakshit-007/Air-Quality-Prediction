"""
alerts.py

Email & optional Twilio SMS alerts.

Configure via environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL_FROM, ALERT_EMAIL_TO
  (Optional) TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO
"""
import os
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject, body):
    host = os.environ.get('SMTP_HOST')
    port = int(os.environ.get('SMTP_PORT', 587))
    user = os.environ.get('SMTP_USER')
    pwd = os.environ.get('SMTP_PASS')
    sender = os.environ.get('ALERT_EMAIL_FROM')
    recipient = os.environ.get('ALERT_EMAIL_TO')

    if not all([host,user,pwd,sender,recipient]):
        raise RuntimeError("SMTP environment variables incomplete. Set SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL_FROM, ALERT_EMAIL_TO")

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient

    s = smtplib.SMTP(host, port, timeout=10)
    try:
        s.starttls()
        s.login(user, pwd)
        s.sendmail(sender, [recipient], msg.as_string())
        print("Email alert sent to", recipient)
    finally:
        s.quit()

def send_sms_twilio(body):
    try:
        from twilio.rest import Client
    except Exception:
        raise RuntimeError("twilio package not installed. pip install twilio to enable SMS alerts.")
    sid = os.environ.get('TWILIO_SID')
    token = os.environ.get('TWILIO_TOKEN')
    from_no = os.environ.get('TWILIO_FROM')
    to_no = os.environ.get('TWILIO_TO')
    if not all([sid,token,from_no,to_no]):
        raise RuntimeError("Twilio env vars missing. Set TWILIO_SID,TWILIO_TOKEN,TWILIO_FROM,TWILIO_TO")
    client = Client(sid, token)
    msg = client.messages.create(body=body, from_=from_no, to=to_no)
    print("SMS sent, sid:", msg.sid)

def check_and_alert(aqi_value, threshold=150.0, notify_methods=('email',)):
    if aqi_value >= threshold:
        subject = f"AQI ALERT: value {aqi_value} >= threshold {threshold}"
        body = f"Air quality has reached {aqi_value} which is >= configured threshold {threshold}."
        for method in notify_methods:
            if method == 'email':
                try:
                    send_email_alert(subject, body)
                except Exception as e:
                    print("Email alert failed:", e)
            elif method == 'sms':
                try:
                    send_sms_twilio(body)
                except Exception as e:
                    print("SMS alert failed:", e)
            else:
                print("Unknown alert method:", method)
    else:
        print("AQI OK:", aqi_value)
