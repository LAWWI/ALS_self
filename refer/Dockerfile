# 1. ใช้ micromamba เป็น base
FROM mambaorg/micromamba:latest

USER root

# 2. ติดตั้ง System Dependencies สำหรับ Spark และ dependencies อื่นๆ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# 3. ตั้งค่า JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ก๊อปปี้ไฟล์ requirements เข้าไปก่อน (ทำในนาม root เพื่อความชัวร์เรื่องสิทธิ์)
COPY req.txt /tmp/req.txt
RUN chown $MAMBA_USER:$MAMBA_USER /tmp/req.txt

USER $MAMBA_USER
WORKDIR /app

# 4. สร้าง Env และติดตั้ง Library จากไฟล์ requirements.txt
# เพิ่มการ clean เพื่อลดขนาด image
RUN micromamba create -n de_env python=3.9 -c conda-forge -y && \
    micromamba run -n de_env pip install --no-cache-dir -r /tmp/req.txt && \
    micromamba clean --all --yes

# 5. สร้าง Jupyter Kernel
RUN micromamba run -n de_env python -m ipykernel install --user --name de_env --display-name "Spark & Recs (Py3.9)"

# 6. ตั้งค่า Auto-activate เมื่อเข้า terminal
RUN echo "micromamba activate de_env" >> ~/.bashrc

# 7. EXPOSE Port
EXPOSE 8888 4040

# แก้ไขคำสั่งรัน: เปลี่ยนจากรอเฉยๆ เป็นรัน Jupyter Notebook ทันที
CMD ["micromamba", "run", "-n", "de_env", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]