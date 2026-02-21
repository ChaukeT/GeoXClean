# Allow running the app with: python -m block_model_viewer

# Setup session logging FIRST (before importing main)
try:
    from block_model_viewer.utils.session_logger import setup_session_logging
    log_file = setup_session_logging()
    print(f"\n{'='*80}")
    print(f"GeoX Session Logs: {log_file}")
    print(f"{'='*80}\n")
except Exception as e:
    print(f"Warning: Session logging setup failed: {e}")
    import logging
    logging.basicConfig(level=logging.INFO)

from .main import main

if __name__ == "__main__":
    main()
