# Deployment Guide for AQI Forecasting System

This project is now configured for production deployment using multiple methods.

## Option 1: Docker (Recommended for Portability)

You can run the entire system (Web App + API) using Docker and Docker Compose.

1.  **Install Docker & Docker Compose** on your server.
2.  **Configure Environment Variables**:
    - Edit the `.env` file in the project root.
    - Set `FLASK_SECRET_KEY` and `WAQI_TOKEN`.
3.  **Build and Start**:
    ```bash
    docker-compose up --build -d
    ```
3.  **Access**:
    - Web App: `http://your-server-ip:5000`
    - Standalone API: `http://your-server-ip:5001`

## Option 2: Platform-as-a-Service (PaaS) - Heroku / Render / Railway

The project includes a `Procfile` and `runtime.txt` for these platforms.

1.  **Connect your GitHub Repository** to the platform.
2.  **Environment Variables**:
    - Set `WAQI_TOKEN` to your WAQI API token.
    - Set `FLASK_ENV=production`.
3.  **Deploy**: The platform will automatically detect the `Procfile` and start the web process.

## Option 3: Traditional VPS Deployment (Ubuntu/Debian)

1.  **Install Dependencies**:
    ```bash
    sudo apt update
    sudo apt install python3-pip python3-venv
    ```
2.  **Setup Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Run with Gunicorn (Background)**:
    ```bash
    gunicorn --bind 0.0.0.0:5000 app:app &
    gunicorn --bind 0.0.0.0:5001 api:app &
    ```
    *(Note: Use `systemd` or `supervisor` for more robust background management)*

## Important Security Notes

- **Secret Keys**: Ensure `app.secret_key` in `app.py` is set to a secure, random value in production (ideally via an environment variable).
- **Debug Mode**: The `app.py` is currently set to `debug=True` in its entry point. In production, use `gunicorn` as specified above, which ignores the `app.run()` block.
