from setuptools import setup, find_packages

setup(
    name="crm-backend",
    version="1.0.0",
    description="AI-Powered Process Mining CRM Backend",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
    ],
    python_requires=">=3.8",
) 