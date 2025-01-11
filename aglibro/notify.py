import yaml
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import argparse

smtp_conf: dict

def report_with_smtp(title, message, msg_from=None):
    try:
        host = smtp_conf['host']
        port = smtp_conf['port']
        username = smtp_conf['username']
        password = smtp_conf['password']
        receiver = smtp_conf.get('receiver', username)
        use_ssl = smtp_conf.get('use_ssl', False)
        assert host != None and port != None and username != None and password != None
        if receiver == None:
            receiver = username

        msg = MIMEText(message, 'plain', 'utf-8')
        msg['Subject'] = title
        msg['From'] = '%s <%s>' % (Header(msg_from, 'utf-8').encode(), username) if msg_from else username
        msg['To'] = Header(receiver if isinstance(receiver, str) else ",".join(receiver), 'utf-8')

        # print(msg)

        try:
            smtp = smtplib.SMTP_SSL(host, port) if use_ssl else smtplib.SMTP(host, port)
            smtp.login(username, password)
            smtp.sendmail(username, receiver, msg.as_string())
        except smtplib.SMTPException:
            print("Error: 发送邮件失败")
            raise
    except KeyError:
        print("Cannot report with SMTP: some secret_key not set")
        return
    except Exception:
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--warning_file", type=str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        smtp_conf = conf['email']
    
    if args.warning_file:
        message = f"命令 {args.warning_file} 运行失败超过五次，请检查！"
    else:
        message = "运行失败超过五次，请检查！"
    report_with_smtp("服务器警告：运行失败", message)