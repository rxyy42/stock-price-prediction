import tkinter as tk
from tkinter import messagebox, scrolledtext
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import requests
import google.generativeai as genai
import os
import threading
import queue
import pickle
import tensorflow as tf
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
LSTM_MODEL_PATH = 'lstm_model.h5' # Assumed pre-trained model
SCALER_PATH = 'scaler.pkl'       # Assumed pre-fitted scaler
LSTM_SEQUENCE_LENGTH = 60        # Match the training sequence length
LSTM_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume'] # Features LSTM model expects

# --- Setup OpenAI ---
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        # Optionally disable chat features if config fails
        GEMINI_API_KEY = None # Indicate config failure
else:
    print("Warning: GEMINI_API_KEY not found in .env file. Gemini chat features disabled.")

# --- Data Fetching Class ---
class StockDataFetcher:
    def fetch_stock_history(self, ticker, period):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                raise ValueError("No historical data found.")
            # Ensure standard columns exist
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                 if col not in data.columns:
                     raise ValueError(f"Required column '{col}' not found in yfinance data.")
            return data
        except Exception as e:
            print(f"Error fetching stock history for {ticker}: {e}")
            raise # Re-raise the exception to be caught later

    def fetch_news(self, ticker):
        if not NEWS_API_KEY:
            print("Warning: NEWS_API_KEY not found. News fetching disabled.")
            return ["News API Key not configured."]

        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&pageSize=5&sortBy=publishedAt&language=en"
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            articles = data.get('articles', [])
            if not articles:
                return ["No recent news found."]
            return [f"{article['title']} - {article['source']['name']}" for article in articles]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for {ticker}: {e}")
            return [f"Error fetching news: {e}"]
        except Exception as e:
            print(f"Error processing news response: {e}")
            return ["Error processing news."]

# --- Data Preprocessing Class ---
class DataPreprocessor:
    def __init__(self, scaler_path=SCALER_PATH):
        self.scaler = self._load_scaler(scaler_path)

    def _load_scaler(self, path):
        try:
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
            if not isinstance(scaler, MinMaxScaler):
                 raise TypeError("Loaded object is not a MinMaxScaler.")
            print("Scaler loaded successfully.")
            return scaler
        except FileNotFoundError:
            print(f"Error: Scaler file not found at {path}. LSTM predictions will likely fail.")
            return None
        except Exception as e:
            print(f"Error loading scaler: {e}")
            return None

    def handle_missing_values(self, data):
        # Simple forward fill for missing values
        return data.ffill().bfill() # Forward fill then back fill for safety

    def scale_data(self, data, features=LSTM_FEATURES):
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Cannot scale data.")
        data_to_scale = data[features].copy()
        scaled_data = self.scaler.transform(data_to_scale)
        # Return as DataFrame for easier handling later if needed
        scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=features)
        return scaled_df

    def inverse_scale_predictions(self, predictions_scaled):
        # Assumes predictions are for the 'Close' price, which is typically
        # the target variable and often the 4th feature (index 3) if using OHLCV.
        # Adjust index if your target variable is different.
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Cannot inverse scale.")

        # Create a dummy array with the same shape as the scaler expects
        num_features = self.scaler.n_features_in_
        dummy_array = np.zeros((len(predictions_scaled), num_features))

        # Find the index of the 'Close' column used during scaling
        try:
            close_index = LSTM_FEATURES.index('Close')
        except ValueError:
             raise ValueError("'Close' feature not found in LSTM_FEATURES list used for scaling.")

        dummy_array[:, close_index] = predictions_scaled.flatten() # Place prediction in the correct column

        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy_array)
        return inversed[:, close_index] # Return only the 'Close' price column

    def create_sequences(self, data, sequence_length=LSTM_SEQUENCE_LENGTH):
        sequences = []
        data_np = data.values # Work with numpy array for efficiency
        for i in range(len(data_np) - sequence_length):
            sequences.append(data_np[i:i + sequence_length])
        return np.array(sequences)

# --- Model Classes ---
class LinearRegressionModel:
    def predict(self, data, days_ahead=30):
        """Predicts future prices using Linear Regression on a time index."""
        data_copy = data.reset_index().copy()
        data_copy['Days'] = np.arange(len(data_copy)) # Time index as feature

        X = np.array(data_copy['Days']).reshape(-1, 1)
        y = np.array(data_copy['Close'])

        model = LinearRegression()
        try:
            model.fit(X, y)
        except Exception as e:
            print(f"Error fitting Linear Regression: {e}")
            return None, None # Indicate failure

        future_days_index = np.arange(len(data_copy), len(data_copy) + days_ahead)
        future_days_feature = future_days_index.reshape(-1, 1)

        try:
            future_prices = model.predict(future_days_feature)
            # Create future dates for plotting alignment
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
            return future_prices, future_dates
        except Exception as e:
            print(f"Error predicting with Linear Regression: {e}")
            return None, None

class LSTMModel:
    def __init__(self, model_path=LSTM_MODEL_PATH):
        self.model = self._load_model(model_path)

    def _load_model(self, path):
        try:
            model = tf.keras.models.load_model(path)
            print("LSTM Model loaded successfully.")
            return model
        except FileNotFoundError:
            print(f"Error: LSTM model file not found at {path}.")
            return None
        except Exception as e: # Catch broader exceptions during loading
            print(f"Error loading LSTM model: {e}")
            return None

    def predict(self, scaled_data, sequence_length=LSTM_SEQUENCE_LENGTH, days_ahead=30):
        """Predicts future prices using the loaded LSTM model."""
        if self.model is None:
            raise ValueError("LSTM model not loaded. Cannot predict.")
        if len(scaled_data) < sequence_length:
             raise ValueError(f"Not enough data ({len(scaled_data)}) for sequence length ({sequence_length}).")

        # Use the last 'sequence_length' days from the historical data
        last_sequence = scaled_data[-sequence_length:].values
        current_batch = last_sequence.reshape(1, sequence_length, scaled_data.shape[1]) # Reshape for LSTM input

        future_predictions_scaled = []

        for _ in range(days_ahead):
            try:
                # Predict next step
                current_pred_scaled = self.model.predict(current_batch)[0] # Get prediction for the next step
                future_predictions_scaled.append(current_pred_scaled)

                # Prepare the next input batch:
                # Remove the first timestep and append the prediction
                # Need to reshape prediction to match feature dimension
                # Assuming prediction output shape is (num_output_features,)
                # If model predicts only 'Close', need to reconstruct the full feature vector
                # --- THIS IS A SIMPLIFICATION ---
                # A more robust approach would predict all features or handle single feature output carefully
                if len(current_pred_scaled) != scaled_data.shape[1]:
                     # Assuming model predicts only 'Close' (index 3)
                     new_row = np.zeros(scaled_data.shape[1])
                     new_row[LSTM_FEATURES.index('Close')] = current_pred_scaled[0] # Assuming single output
                     # Use previous step's other features (simplification)
                     for i in range(scaled_data.shape[1]):
                         if i != LSTM_FEATURES.index('Close'):
                             new_row[i] = current_batch[0, -1, i] # Take from last known step
                else:
                    new_row = current_pred_scaled # Model predicts all features

                new_batch_entry = new_row.reshape(1, 1, scaled_data.shape[1])
                current_batch = np.append(current_batch[:, 1:, :], new_batch_entry, axis=1)

            except Exception as e:
                print(f"Error during LSTM prediction loop: {e}")
                return None # Indicate failure

        return np.array(future_predictions_scaled)

# --- Analysis Orchestrator ---
class StockAnalyzer:
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.preprocessor = DataPreprocessor()
        self.lr_model = LinearRegressionModel()
        self.lstm_model = LSTMModel()

    def analyze(self, ticker, period, days_ahead=30):
        results = {'error': None}
        try:
            # 1. Fetch Data
            hist_data = self.fetcher.fetch_stock_history(ticker, period)
            results['historical_data'] = hist_data

            # 2. Fetch News (can happen in parallel conceptually)
            news = self.fetcher.fetch_news(ticker)
            results['news'] = news

            # 3. Preprocess
            data_clean = self.preprocessor.handle_missing_values(hist_data)
            results['cleaned_data'] = data_clean # Store cleaned data for SMA etc.

            # 4. Linear Regression Prediction
            lr_preds, lr_dates = self.lr_model.predict(data_clean, days_ahead)
            results['lr_predictions'] = lr_preds
            results['lr_dates'] = lr_dates

            # 5. LSTM Prediction
            lstm_preds_scaled = None
            lstm_preds = None
            lstm_dates = lr_dates # Assume same dates for simplicity of plotting
            if self.preprocessor.scaler and self.lstm_model.model:
                try:
                    scaled_data = self.preprocessor.scale_data(data_clean)
                    lstm_preds_scaled = self.lstm_model.predict(scaled_data, days_ahead=days_ahead)
                    if lstm_preds_scaled is not None:
                         # Inverse scale LSTM predictions (assuming 'Close' price prediction)
                         lstm_preds = self.preprocessor.inverse_scale_predictions(lstm_preds_scaled)
                except Exception as e:
                    print(f"Error during LSTM processing: {e}")
                    results['lstm_error'] = str(e) # Report LSTM specific error
            else:
                 results['lstm_error'] = "LSTM Model or Scaler not loaded."

            results['lstm_predictions'] = lstm_preds
            results['lstm_dates'] = lstm_dates # Use same dates as LR for plotting

            # 6. Generate Recommendation (Simple Example - Needs Refinement)
            recommendation = self._generate_recommendation(data_clean, lr_preds, lstm_preds)
            results['recommendation'] = recommendation

            # 7. Calculate SMA
            data_with_sma = self._calculate_moving_average(data_clean)
            results['data_with_sma'] = data_with_sma


        except Exception as e:
            print(f"Analysis failed for {ticker}: {e}")
            results['error'] = str(e)

        return results

    def _calculate_moving_average(self, data, window=30):
        data = data.copy()
        data['SMA'] = data['Close'].rolling(window=window, min_periods=1).mean()
        return data

    def _generate_recommendation(self, data, lr_future_prices, lstm_future_prices):
        # --- This logic needs significant refinement based on strategy ---
        # Example: Use LSTM if available, otherwise LR. More complex logic possible.
        current_price = data['Close'].iloc[-1]
        sma = data['SMA'].iloc[-1] if 'SMA' in data else data['Close'].rolling(30).mean().iloc[-1] # Calculate if needed

        predicted_price = None
        source = "N/A"

        if lstm_future_prices is not None and len(lstm_future_prices) > 0:
            predicted_price = lstm_future_prices[-1]
            source = "LSTM"
        elif lr_future_prices is not None and len(lr_future_prices) > 0:
            predicted_price = lr_future_prices[-1]
            source = "LR"

        if predicted_price is None:
            return f"Hold (Prediction N/A)"

        recommendation = "Hold"
        if current_price > sma and predicted_price > current_price * 1.03: # Adjusted threshold
            recommendation = "Buy"
        elif current_price < sma and predicted_price < current_price * 0.97: # Adjusted threshold
            recommendation = "Sell"

        return f"{recommendation} (Based on {source})"


# --- GUI Application Class ---
class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Analysis Tool (LR + LSTM Hybrid)")
        self.root.geometry("800x750") # Adjusted size

        self.analyzer = StockAnalyzer()
        self.analysis_queue = queue.Queue()
        self.chat_queue = queue.Queue()

        # --- UI Elements ---
        # Input Frame
        input_frame = tk.Frame(root, pady=10)
        input_frame.pack(fill=tk.X)

        tk.Label(input_frame, text="Ticker:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.entry_ticker = tk.Entry(input_frame, width=10)
        self.entry_ticker.grid(row=0, column=1, padx=5, pady=2)

        tk.Label(input_frame, text="Period (e.g., 1y, 6mo):").grid(row=0, column=2, padx=5, pady=2, sticky='w')
        self.entry_period = tk.Entry(input_frame, width=10)
        self.entry_period.insert(0, "1y") # Default value
        self.entry_period.grid(row=0, column=3, padx=5, pady=2)

        tk.Label(input_frame, text="Stop-Loss:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.entry_stop_loss = tk.Entry(input_frame, width=10)
        self.entry_stop_loss.grid(row=1, column=1, padx=5, pady=2)

        tk.Label(input_frame, text="Target Price:").grid(row=1, column=2, padx=5, pady=2, sticky='w')
        self.entry_target_price = tk.Entry(input_frame, width=10)
        self.entry_target_price.grid(row=1, column=3, padx=5, pady=2)

        self.analyze_button = tk.Button(input_frame, text="Analyze Stock", command=self._start_analysis)
        self.analyze_button.grid(row=0, column=4, rowspan=2, padx=10, pady=5, sticky='ns')

        self.status_label = tk.Label(input_frame, text="", fg="blue")
        self.status_label.grid(row=2, column=0, columnspan=5, pady=2)

        # Plot Frame
        plot_frame = tk.Frame(root, relief=tk.SUNKEN, borderwidth=1)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4)) # Create figure and axes
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.ax.grid(True)
        self.ax.set_title("Stock Analysis Plot")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price (USD)")
        self.fig.tight_layout()

        # News and Chat Frame
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        # News Section
        news_frame = tk.LabelFrame(bottom_frame, text="Latest News", padx=5, pady=5)
        news_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.news_text = scrolledtext.ScrolledText(news_frame, wrap=tk.WORD, height=6, width=50)
        self.news_text.pack(fill=tk.BOTH, expand=True)
        self.news_text.insert(tk.END, "News headlines will appear here...")
        self.news_text.config(state=tk.DISABLED)

        # Chat Section
        chat_frame = tk.LabelFrame(bottom_frame, text="Ask Gemini", padx=5, pady=5)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.entry_chat = tk.Entry(chat_frame, width=40)
        self.entry_chat.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        self.chat_button = tk.Button(chat_frame, text="Ask", command=self._start_chat_query)
        self.chat_button.pack(side=tk.TOP, pady=(0, 5))

        self.chat_response_text = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=4, width=40)
        self.chat_response_text.pack(fill=tk.BOTH, expand=True)
        self.chat_response_text.insert(tk.END, "Gemini response...")
        self.chat_response_text.config(state=tk.DISABLED)

    def _start_analysis(self):
        ticker = self.entry_ticker.get().upper()
        period = self.entry_period.get().lower()

        if not ticker or not period:
            messagebox.showerror("Input Error", "Please provide both stock ticker and period.")
            return

        # Store stop-loss/target for later use in UI thread
        try:
            self.stop_loss = float(self.entry_stop_loss.get()) if self.entry_stop_loss.get() else None
            self.target_price = float(self.entry_target_price.get()) if self.entry_target_price.get() else None
        except ValueError:
            messagebox.showerror("Input Error", "Stop-loss and target price must be numeric.")
            return

        self.analyze_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"Analyzing {ticker}...")
        self.clear_plot()
        self.news_text.config(state=tk.NORMAL)
        self.news_text.delete(1.0, tk.END)
        self.news_text.insert(tk.END, "Fetching data...")
        self.news_text.config(state=tk.DISABLED)

        # Run analysis in a separate thread
        self.analysis_thread = threading.Thread(target=self._analysis_worker, args=(ticker, period), daemon=True)
        self.analysis_thread.start()
        self.root.after(100, self._check_analysis_queue) # Start checking the queue

    def _analysis_worker(self, ticker, period):
        """Worker function to run analysis in background thread."""
        results = self.analyzer.analyze(ticker, period)
        self.analysis_queue.put(results) # Put results in queue for main thread

    def _check_analysis_queue(self):
        """Check the queue for results from the analysis thread."""
        try:
            results = self.analysis_queue.get_nowait()
            self._update_ui(results)
        except queue.Empty:
            # If queue is empty, check again later if thread is alive
            if self.analysis_thread.is_alive():
                self.root.after(100, self._check_analysis_queue)
            # else: # Thread finished but queue was empty (shouldn't happen if worker always puts result)
            #     self.status_label.config(text="Analysis finished.")
            #     self.analyze_button.config(state=tk.NORMAL)

    def _update_ui(self, results):
        """Update the GUI with results - MUST run in main thread."""
        self.analyze_button.config(state=tk.NORMAL)
        self.status_label.config(text="Analysis complete.")

        if results.get('error'):
            messagebox.showerror("Analysis Error", results['error'])
            self.status_label.config(text=f"Error: {results['error']}")
            return

        # Update News
        self.news_text.config(state=tk.NORMAL)
        self.news_text.delete(1.0, tk.END)
        self.news_text.insert(tk.END, "\n".join(results.get('news', ["No news available."])))
        self.news_text.config(state=tk.DISABLED)

        # Update Plot
        self._plot_data(results)

        # Check Alerts (using cleaned data)
        current_price = results.get('cleaned_data')['Close'].iloc[-1] if results.get('cleaned_data') is not None else None
        if current_price:
            if self.stop_loss is not None and current_price <= self.stop_loss:
                messagebox.showwarning("Alert", f"Price ({current_price:.2f}) hit stop-loss ({self.stop_loss:.2f}). Consider selling!")
            elif self.target_price is not None and current_price >= self.target_price:
                messagebox.showinfo("Alert", f"Price ({current_price:.2f}) hit target price ({self.target_price:.2f}). Consider taking profit!")

    def clear_plot(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title("Stock Analysis Plot")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price (USD)")
        self.canvas.draw()

    def _plot_data(self, results):
        self.clear_plot() # Clear previous plot

        ticker = self.entry_ticker.get().upper()
        period = self.entry_period.get().lower()
        data = results.get('data_with_sma') # Use data with SMA calculated
        lr_preds = results.get('lr_predictions')
        lr_dates = results.get('lr_dates')
        lstm_preds = results.get('lstm_predictions')
        lstm_dates = results.get('lstm_dates') # Should be same as lr_dates
        recommendation = results.get('recommendation', 'N/A')

        if data is None or data.empty:
            self.ax.text(0.5, 0.5, 'No data to plot', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        # Plot historical data and SMA
        self.ax.plot(data.index, data['Close'], label=f'{ticker} Close ({period})', color='blue', marker='.', markersize=4, linestyle='-')
        if 'SMA' in data:
            self.ax.plot(data.index, data['SMA'], label=f'30-Day SMA', color='orange', linestyle='--')

        # Plot Linear Regression predictions
        if lr_preds is not None and lr_dates is not None:
            self.ax.plot(lr_dates, lr_preds, label=f'LR Prediction', color='green', linestyle='--', marker='x')
            # Annotate last LR prediction
            self.ax.annotate(f'LR: ${lr_preds[-1]:.2f}', (lr_dates[-1], lr_preds[-1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='green')


        # Plot LSTM predictions
        if results.get('lstm_error'):
             print(f"LSTM Plotting skipped: {results['lstm_error']}")
             # Optionally display LSTM error on plot
             self.ax.text(0.95, 0.95, f"LSTM Err: {results['lstm_error'][:30]}...", color='red', fontsize=8, ha='right', va='top', transform=self.ax.transAxes)
        elif lstm_preds is not None and lstm_dates is not None:
            self.ax.plot(lstm_dates, lstm_preds, label=f'LSTM Prediction', color='red', linestyle=':', marker='o', markersize=3)
             # Annotate last LSTM prediction
            self.ax.annotate(f'LSTM: ${lstm_preds[-1]:.2f}', (lstm_dates[-1], lstm_preds[-1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='red')
        else:
             print("LSTM predictions not available for plotting.")


        # Annotate current price and recommendation
        current_price = data['Close'].iloc[-1]
        self.ax.annotate(f'Current: ${current_price:.2f}', (data.index[-1], current_price), textcoords="offset points", xytext=(0,10), ha='right', fontsize=9, color='blue')
        self.ax.annotate(f'Rec: {recommendation}', (data.index[len(data)//2], self.ax.get_ylim()[0]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5, pad=0.2))


        self.ax.set_title(f'{ticker} Analysis ({period}) - Rec: {recommendation}')
        self.ax.legend(fontsize='small')
        self.fig.autofmt_xdate() # Rotate dates for better visibility
        self.fig.tight_layout()
        self.canvas.draw() # Redraw the canvas


    # --- Gemini Integration ---
    def _start_chat_query(self):
        """Initiates the Gemini query process."""
        user_question = self.entry_chat.get()
        if not user_question:
            messagebox.showerror("Input Error", "Please enter a question for the chatbot.")
            return

        # Check if Gemini API Key is configured (loaded at script start)
        if not GEMINI_API_KEY:
             messagebox.showerror("API Error", "Gemini API Key not configured. Cannot query chatbot.")
             return

        # Disable button and update status
        self.chat_button.config(state=tk.DISABLED)
        self.chat_response_text.config(state=tk.NORMAL)
        self.chat_response_text.delete(1.0, tk.END)
        self.chat_response_text.insert(tk.END, "Asking Gemini...") # Updated status
        self.chat_response_text.config(state=tk.DISABLED)

        # Run chat query in a separate thread
        self.chat_thread = threading.Thread(target=self._chat_worker, args=(user_question,), daemon=True)
        self.chat_thread.start()
        # Start checking the queue for the response
        self.root.after(100, self._check_chat_queue)

    def _chat_worker(self, question):
        """Worker function for Gemini query, runs in a separate thread."""
        try:
            # 1. Select the Gemini model
            # Use 'gemini-pro' for general tasks or 'gemini-1.5-pro-latest' etc.
            model = genai.GenerativeModel('gemini-2.0-flash')

            # 2. Generate content
            # For more context, you could modify the prompt:
            prompt = f"You are a helpful financial assistant. Answer this question about finance precisely in 2-3 sentences: {question}"
            response = model.generate_content(prompt)
            # response = model.generate_content(question) # Using direct question as prompt

            # 3. Extract the text, checking for safety blocks
            if response.parts:
                 answer = response.text
            else:
                 # Handle cases where the response might be blocked
                 answer = "Response could not be generated (check safety settings or prompt)."
                 # Log the full response for debugging if needed
                 print(f"Gemini response blocked or empty. Response object: {response}")

            # Put the result into the queue for the main thread
            self.chat_queue.put({'answer': answer})

        except Exception as e:
            # Catch potential errors during API call or processing
            error_message = f"Gemini Error: {str(e)}"
            # Log the error
            print(error_message)
            # Put the error into the queue
            self.chat_queue.put({'error': error_message})

    def _check_chat_queue(self):
        """Checks the queue for results from the Gemini worker thread."""
        try:
            # Try to get a result from the queue without blocking
            result = self.chat_queue.get_nowait()
            # If successful, update the UI
            self._update_chat_ui(result)
        except queue.Empty:
            # If the queue is empty, check if the thread is still running
            if self.chat_thread.is_alive():
                # If still running, schedule another check shortly
                self.root.after(100, self._check_chat_queue)
            # If thread finished and queue is empty, do nothing (shouldn't normally happen)

    def _update_chat_ui(self, result):
        """Updates the chat response area in the main Tkinter thread."""
        # Re-enable the Ask button
        self.chat_button.config(state=tk.NORMAL)
        # Enable text area for modification
        self.chat_response_text.config(state=tk.NORMAL)
        # Clear previous content
        self.chat_response_text.delete(1.0, tk.END)

        # Check if the result dictionary contains an error
        if result.get('error'):
            self.chat_response_text.insert(tk.END, f"Error: {result['error']}")
        else:
            # Get the answer, provide default if missing
            answer = result.get('answer', "No response received.")
            self.chat_response_text.insert(tk.END, answer)

        # Disable text area again to make it read-only
        self.chat_response_text.config(state=tk.DISABLED)


# --- Main Execution ---
if __name__ == "__main__":
    # Check if essential files exist before starting GUI
    if not os.path.exists(LSTM_MODEL_PATH):
         print(f"FATAL ERROR: LSTM Model file '{LSTM_MODEL_PATH}' not found.")
         # Optionally show a GUI error message here too
         # exit() # Or allow GUI to load but show error prominently
    if not os.path.exists(SCALER_PATH):
         print(f"FATAL ERROR: Scaler file '{SCALER_PATH}' not found.")
         # exit()

    root = tk.Tk()
    app = StockAnalysisApp(root)
    root.mainloop()