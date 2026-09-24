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
    
    print("Starting FastAPI Server at http://localhost:8000 ...")
    
    try:
        import uvicorn
        webbrowser.open("http://localhost:8000")
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    except ImportError:
        print("Error: Dependencies missing. Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
