import os
import webbrowser


def main():
    print("="*65)
    print(" DERMATOBLOOD AI - SIFT + CNN CLINICAL SYSTEM V5.0")
    print("="*65)

    models_path = os.path.join(os.path.dirname(__file__), "models", "blood_group_model.pth")
    if not os.path.exists(models_path):
        print("Pre-trained model weights not found. Initializing training...")
        from scripts.generate_data_and_train import train_and_save_model
        train_and_save_model()

    port = int(os.environ.get("PORT", "8001"))
    url = f"http://127.0.0.1:{port}"
    print(f"Starting FastAPI Server at {url} ...")
    
    try:
        import uvicorn
        webbrowser.open(url)
        uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=False)
    except ImportError:
        print("Error: Dependencies missing. Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
