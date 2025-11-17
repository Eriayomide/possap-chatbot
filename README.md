# POSSAP AI Chatbot

An intelligent chatbot assistant for the Police Specialized Services Automation Project (POSSAP) platform in Nigeria. This AI-powered chatbot helps users with police service applications, payment inquiries, character certificate requests, and other POSSAP-related queries.

## 🚔 Features

- **Personalized Conversations**: Recognizes and remembers user names during the session
- **POSSAP Knowledge Base**: Trained on official POSSAP procedures and FAQs
- **Smart Hyperlinks**: Automatically converts emails and websites to clickable links
- **Quick Actions**: Pre-built buttons for common queries
- **Real-time Responses**: Powered by advanced AI models for instant assistance
- **Vector Search**: Uses ChromaDB and sentence transformers for relevant FAQ retrieval
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Custom Avatar**: Features "Amina" - your friendly POSSAP assistant

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **AI Language Model** - Advanced chatbot capabilities
- **ChromaDB** - Vector database for FAQ storage
- **SentenceTransformers** - Text embeddings for semantic search
- **Flask-CORS** - Cross-origin resource sharing

### Frontend
- **HTML5/CSS3** - Modern responsive design
- **Vanilla JavaScript** - Pure JS for optimal performance
- **CSS Animations** - Smooth user experience
- **Custom UI** - POSSAP-branded interface with Amina avatar

### DevOps
- **Docker** - Containerized deployment
- **Git** - Version control

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- API key for AI model (if required)
- Git
- Docker (optional, for containerized deployment)

### Local Development

1. **Clone the repository**
```bash
   git clone https://github.com/Eriayomide/possap-chatbot.git
   cd possap-chatbot
```

2. **Set up the backend**
```bash
   cd posap_backend
   pip install -r requirements.txt
```

3. **Create environment file**
```bash
   # Create .env file in posap_backend folder
   echo "API_KEY=your_api_key_here" > .env
```

4. **Run the backend server**
```bash
   python possap_chatbot.py
```
   Backend will be available at: `http://localhost:5000`

5. **Open the frontend**
```bash
   # In a new terminal, from project root
   cd posap_frontend
   # Open possap_production.html in your browser or use a local server
   python -m http.server 8000
```
   Frontend will be available at: `http://localhost:8000`

### Docker Deployment (Recommended)

1. **Build the Docker image**
```bash
   docker build -t possap-chatbot .
```

2. **Run the container**
```bash
   docker run -p 5000:5000 -e API_KEY=your_api_key_here possap-chatbot
```

## 📚 Knowledge Base

The chatbot is trained on official POSSAP services and procedures covering:

- **Character Certificate**: Application process, requirements, status tracking
- **Police Escort Services**: Booking, pricing, approval process
- **Guard Services**: Request procedures, billing information
- **Special Duties**: Application for special police assignments
- **Payment Issues**: Failed payments, refunds, invoice queries
- **Technical Support**: Login issues, portal errors, account problems
- **Document Requirements**: Identification, proof of address, supporting documents
- **Processing Times**: Expected timelines for various services
- **Contact Information**: Support channels and office locations

## 📱 Usage Examples

### Sample Conversations

**User**: "Hi, I'm Adebayo"  
**Amina**: "Hello Adebayo! Welcome to POSSAP. I'm Amina, your virtual assistant. How can I help you today with our police services?"

**User**: "How do I apply for a character certificate?"  
**Amina**: "Great question, Adebayo! To apply for a Police Character Certificate through POSSAP:

1. Visit www.possap.gov.ng
2. Create an account or login
3. Select 'Character Certificate' from services
4. Fill in your personal details
5. Upload required documents (valid ID, passport photo)
6. Make payment
7. Submit application

Processing typically takes 2-3 weeks. You'll receive email updates on your application status."

## 🤝 Contributing

We welcome contributions to improve the POSSAP chatbot!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For technical support or questions:

- **POSSAP Support**: support@possap.gov.ng
- **Payment Issues**: payments@possap.gov.ng
- **General Inquiries**: info@possap.gov.ng
- **Website**: www.possap.gov.ng

## 🚀 Deployment

### Render (Recommended)

**Backend Deployment:**
1. Create new Web Service on Render
2. Connect your GitHub repository
3. Set root directory to `posap_backend`
4. Add environment variables (`API_KEY`)

**Frontend Deployment:**
1. Create new Static Site on Render
2. Set root directory to `posap_frontend`

---

**Made with ❤️ for Nigeria Police Force POSSAP** 🇳🇬 🚔