from flask import Flask, request, jsonify, send_from_directory, session, send_file
import os
from anthropic import Anthropic
from flask_cors import CORS
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import uuid
import re
import time
from threading import Lock

# Load environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key loaded: {'Yes' if anthropic_api_key else 'No'}")
client = Anthropic(api_key=anthropic_api_key)
app = Flask(__name__)
# Secret key for Flask sessions
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "dev-secret"  # fallback for local dev
)
CORS(app)


# In-memory conversation store
conversations = {}
conversations_lock = Lock()

class ConversationManager:
    """Manage conversation state including user names and message history"""
    
    def __init__(self):
        self.conversations = {}
        self.lock = Lock()
        
    def get_or_create_conversation(self, conversation_id: str) -> Dict:
        """Get or create a conversation"""
        with self.lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = {
                    'user_name': None,
                    'created_at': time.time(),
                    'last_activity': time.time(),
                    'messages': []
                }
            else:
                self.conversations[conversation_id]['last_activity'] = time.time()
            return self.conversations[conversation_id]
    
    def set_user_name(self, conversation_id: str, name: str):
        """Set user name for a conversation"""
        with self.lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id]['user_name'] = name
                self.conversations[conversation_id]['last_activity'] = time.time()
    
    def get_user_name(self, conversation_id: str) -> str:
        """Get user name for a conversation"""
        with self.lock:
            conv = self.conversations.get(conversation_id)
            return conv['user_name'] if conv else None
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add a message to conversation history"""
        with self.lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id]['messages'].append({
                    'role': role,
                    'content': content,
                    'timestamp': time.time()
                })
                # Keep only last 10 messages to avoid token limits
                if len(self.conversations[conversation_id]['messages']) > 10:
                    self.conversations[conversation_id]['messages'] = \
                        self.conversations[conversation_id]['messages'][-10:]
                self.conversations[conversation_id]['last_activity'] = time.time()
    
    def get_conversation_history(self, conversation_id: str, max_messages: int = 10) -> List[Dict]:
        """Get conversation history"""
        with self.lock:
            conv = self.conversations.get(conversation_id)
            if conv and 'messages' in conv:
                return conv['messages'][-max_messages:]
            return []
    
    def get_full_conversation(self, conversation_id: str) -> Dict:
        """Get full conversation data including all messages"""
        with self.lock:
            conv = self.conversations.get(conversation_id)
            if conv:
                return {
                    'user_name': conv.get('user_name'),
                    'messages': conv.get('messages', []),
                    'created_at': conv.get('created_at'),
                    'last_activity': conv.get('last_activity')
                }
            return None
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up conversations older than max_age_hours"""
        with self.lock:
            current_time = time.time()
            to_remove = []
            for conv_id, conv_data in self.conversations.items():
                if current_time - conv_data['last_activity'] > max_age_hours * 3600:
                    to_remove.append(conv_id)
            
            for conv_id in to_remove:
                del self.conversations[conv_id]

# Initialize the ConversationManager
conversation_manager = ConversationManager()

# Initialize SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# POSSAP Knowledge Base - UPDATED FAQs from Revised Official Document
possap_faqs = [
    # Registration and Account Creation (Q1-Q4, Q22-Q23)
    {
        "question": 'I tried to use my NIN/BVN to sign up on the POSSAP portal and got an error saying "something went wrong, please contact POSSAP admin"',
        "answer": 'This means your NIN or BVN record does not have a phone number linked to it. If using NIN, visit the nearest NIMC office to update your record with your current phone number. If using BVN, visit your bank to update your phone number in your BVN details. After updating, contact POSSAP to have your information revalidated in the system.',
        "category": 'registration'
    },
    {
        "question": 'The name arrangement I see on the POSSAP site is different from what appears on my passport',
        "answer": 'POSSAP pulls your name directly from NIMC or your bank. Just visit your nearest NIMC office or your bank branch to update how your name appears on your BVN/NIN and contact POSSAP at info@possap.gov.ng for revalidation to proceed with your application.',
        "category": 'registration'
    },
    {
        "question": 'The phone number or email shown on the POSSAP site is my old one. How can I change it?',
        "answer": 'POSSAP retrieves information such as your name, phone number etc. directly from the National Identify Management Commission NIMC or Nigeria Inter-bank Settlement System (NIBSS). If your name appears incorrectly, kindly visit the NIMC head office in your state of residence, or the nearest branch of your bank to update the details of your National identification Number (NIN) or your Bank verification Number (BVN) respectively, reach out to POSSAP with your NIN and updated information for revalidation, before continuing your registration.',
        "category": 'registration'
    },
    {
        "question": "I'm unable to verify my account even after receiving multiple verification codes",
        "answer": 'Please contact POSSAP Customer Service with your NIN/BVN, phone number and email address via: Phone: 02018884040 and/or email: info@possap.gov.ng',
        "category": 'registration'
    },
    {
        "question": 'I did not receive the OTP for account verification. What should I do?',
        "answer": 'Check your spam or junk folder. If not received, confirm your email address is correct and click Resend OTP.',
        "category": 'registration'
    },
    {
        "question": 'The system says "User already exists" during registration. What should I do?',
        "answer": 'This means an account is already linked to that identifier or email. Use the Forgot Password option to regain access.',
        "category": 'registration'
    },
    
    # Tinted Glass Permit Related Issues (Q5-Q7, Q18-Q19, Q24)
    {
        "question": "I made payment for VVS and was debited, but it didn't reflect on my invoice",
        "answer": 'Kindly contact POSSAP Customer Care with your invoice number, Vehicle Identification Number (VIN) and Proof of payment for assistance via the following contact information: Phone: 02018884040 and/or email: info@possap.gov.ng',
        "category": 'tinted_glass'
    },
    {
        "question": 'What document should I upload to support my Tinted Glass Permit (health-related) application?',
        "answer": 'You are required to upload a medical report from a government recognized hospital which is duly signed and stamped by the hospital to support your health claim when submitting your application on the POSSAP portal.',
        "category": 'tinted_glass'
    },
    {
        "question": 'What documents do I need to upload as an applicant for the Tinted Glass Permit opting for the Virtual verification?',
        "answer": 'Required Documents for uploading include: Proof of ownership of vehicle, Vehicle licensed data page, Supporting document for your reason for application (medical report for health reasons, ID Card for Security reasons, and document proving vehicle is factory fitted with tinted windows for Factory Fitted options).',
        "category": 'tinted_glass'
    },
    {
        "question": 'How long is a Tinted Glass Permit valid?',
        "answer": 'The Tinted Glass Permit is valid for one year from the date of issuance and must be renewed after expiration.',
        "category": 'tinted_glass'
    },
    {
        "question": 'Who is eligible to apply for the Virtual Vehicle Verification System Tinted Glass Permit on the POSSAP platform?',
        "answer": 'Only owners of vehicles with a valid 17-digit Vehicle Identification Number (VIN) that conforms to international standards are eligible for the Virtual Vehicle Verification System.',
        "category": 'tinted_glass'
    },
    {
        "question": 'Why am I redirected to another site for Vehicle Verification, what is the Vehicle verification System about?',
        "answer": "The Vehicle Verification System (VVS) is an external platform integrated with POSSAP. It serves as a Global Vehicle Identification Number (VIN) database that POSSAP utilizes to securely and in real time retrieve comprehensive vehicle information from the global database, thereby ensuring accurate capture of applicants' vehicle details.",
        "category": 'tinted_glass'
    },
    
    # Police Character Certificate (Q8, Q17)
    {
        "question": "I am applying from the diaspora. What proof should I upload to show I'm not in Nigeria?",
        "answer": 'You can upload any valid supporting document, such as: Official Diaspora Proof of residence document, Utility bills (water or electricity), Bank statement, Lease agreement or other proof of residence abroad, Drivers license, Work permit.',
        "category": 'character_certificate'
    },
    {
        "question": 'How much does biometric capturing cost for Police Character Certificate & Tinted Glass Permit?',
        "answer": 'Biometric capturing and physical inspection sessions required for the issuance of Police Character Certificates and Tinted Glass Permits are completely free of charge. Applicants are not required to make any payments for these processes.',
        "category": 'character_certificate'
    },
    
    # Facial Verification Issues (Q9, Q21)
    {
        "question": "The facial verification process isn't capturing my face after several attempts",
        "answer": 'Try the following: Use a Computer (Desktop/Laptop) instead of a mobile device, Ensure adequate lighting in the room of capture, Use your most recent passport photo for upload. If the issue persists, contact POSSAP Customer Service: Phone: 02018884040 and/or email: info@possap.gov.ng',
        "category": 'verification'
    },
    {
        "question": 'Why am I unable to complete the virtual verification, and why do I keep getting an error that says, "Face does not match"?',
        "answer": 'This issue may be due to the use of an outdated passport photograph during your application. Kindly contact the POSSAP Support Team via Phone: 02018884040 or Email: info@possap.gov.ng to request that your uploaded photograph be updated with a more recent one.',
        "category": 'verification'
    },
    
    # Payment-Related Issues (Q10-Q13, Q16, Q25)
    {
        "question": 'Can I make payment for a diaspora application in Naira?',
        "answer": 'Diaspora payments must be made in dollars (or the corresponding currency of your host country), and for the exact amount displayed on the POSSAP portal.',
        "category": 'payment'
    },
    {
        "question": 'I erroneously made double payment on the same invoice. How do I get a refund?',
        "answer": 'Email POSSAP Customer Service at info@possap.gov.ng or call 02018884040, providing the following: Receipt of both payments, Date of payment, Invoice number, Account number paid into, Your Bank details (Account name & Number), Your email address and phone number. Refund processing will follow once verification processes have been concluded.',
        "category": 'payment'
    },
    {
        "question": 'I made payment on the wrong invoice number. Can I transfer the payment to the correct one?',
        "answer": 'Payments cannot be transferred between invoices. You may either use the service linked to the paid invoice or make a new payment under the correct invoice.',
        "category": 'payment'
    },
    {
        "question": 'Can I transfer a payment made under the wrong invoice?',
        "answer": 'No. Kindly note that Payments are linked to specific invoices and cannot be transferred. You will be required to initiate a new payment for the appropriate service.',
        "category": 'payment'
    },
    {
        "question": "I made payment on my generated invoice, but it didn't reflect",
        "answer": "Contact POSSAP Customer Service with your payment receipt and invoice number: Phone: 02018884040 and/or email: info@possap.gov.ng. If payment hasn't reflected on POSSAP's end, you will be advised to contact your bank to lodge a complaint.",
        "category": 'payment'
    },
    {
        "question": 'Why is my payment not reflected after I mistakenly paid in naira instead of 53.76 USD?',
        "answer": 'Payment for the diaspora can only be in US Dollars. If you have made payment in Naira, kindly send an email to info@possap.gov.ng to get a refund and make payment in the correct currency.',
        "category": 'payment'
    },
    
    # Application Status (Q20)
    {
        "question": 'My application has been pending for over 2 weeks now, what do I do to get it approved?',
        "answer": 'Kindly reach out to the POSSAP support team via Phone: 02018884040 and/or email: info@possap.gov.ng with your invoice number or file number to get clarification and resolution on the issue.',
        "category": 'application_status'
    }
]

class HyperlinkProcessor:
    """Class to handle hyperlink processing for POSSAP responses"""
    
    @staticmethod
    def convert_to_hyperlinks(text: str) -> str:
        """Convert URLs and email addresses to HTML hyperlinks"""
        # Use placeholders to prevent nested conversions
        placeholders = {}
        placeholder_counter = [0]
        
        def create_placeholder(content):
            placeholder = f"___PLACEHOLDER_{placeholder_counter[0]}___"
            placeholders[placeholder] = content
            placeholder_counter[0] += 1
            return placeholder
        
        # STEP 1: Convert email addresses to mailto links with placeholders
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        
        def email_replacer(match):
            email = match.group(1)
            link = f'<a href="mailto:{email}" style="color: #0066cc; text-decoration: underline; font-weight: 500;">{email}</a>'
            return create_placeholder(link)
        
        result = re.sub(email_pattern, email_replacer, text)
        
        # STEP 2: Convert URLs to hyperlinks (emails are now placeholders, so won't be affected)
        url_pattern = r'((?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)'
        
        def url_replacer(match):
            url = match.group(1)
            # Skip if it's a placeholder
            if '___PLACEHOLDER_' in url:
                return url
            
            # Handle specific domain mappings
            href = url
            if not url.startswith('http'):
                if 'www.possap.gov.ng' in url:
                    href = url.replace('www.possap.gov.ng', 'https://possap.gov.ng')
                elif url.startswith('www.'):
                    href = f'https://{url[4:]}'
                else:
                    href = f'https://{url}'
            
            link = f'<a href="{href}" target="_blank" rel="noopener noreferrer" style="color: #0066cc; text-decoration: underline; font-weight: 500;">{url}</a>'
            return create_placeholder(link)
        
        result = re.sub(url_pattern, url_replacer, result)
        
        # STEP 3: Replace placeholders with actual HTML
        for placeholder, content in placeholders.items():
            result = result.replace(placeholder, content)
        
        return result
    
    @staticmethod
    def process_faq_answer(answer: str) -> str:
        """Process FAQ answer to include hyperlinks"""
        return HyperlinkProcessor.convert_to_hyperlinks(answer)

class POSSAPRAGSystem:
    def __init__(self):
        self.collection_name = "possap_faqs"
        # Initialize ChromaDB client as instance attribute
        self.chroma_client = chromadb.Client()
        self.hyperlink_processor = HyperlinkProcessor()
        self.setup_vector_database()
    
    def setup_vector_database(self):
        """Initialize ChromaDB collection with POSSAP FAQs"""
        try:
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
            except:
                pass
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Prepare documents for embedding
            documents = []
            metadatas = []
            ids = []
            
            for i, faq in enumerate(possap_faqs):
                # Combine question and answer for better context
                doc_text = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
                documents.append(doc_text)
                metadatas.append({
                    "category": faq['category'],
                    "question": faq['question'],
                    "answer": faq['answer']
                })
                ids.append(str(uuid.uuid4()))
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Vector database initialized with {len(possap_faqs)} FAQs")
            
        except Exception as e:
            print(f"❌ Error setting up vector database: {e}")
    
    def retrieve_relevant_faqs(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve most relevant FAQs based on user query"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            relevant_faqs = []
            if results['metadatas'] and len(results['metadatas'][0]) > 0:
                for metadata in results['metadatas'][0]:
                    relevant_faqs.append({
                        "question": metadata['question'],
                        "answer": metadata['answer'],
                        "category": metadata['category']
                    })
            
            return relevant_faqs
            
        except Exception as e:
            print(f"❌ Error retrieving FAQs: {e}")
            return []
    
    def generate_rag_response(self, user_query: str, user_name: str = None, conversation_history: List[Dict] = None) -> Dict:
        """Generate response using RAG with conversation context"""
        try:
            # Step 1: Retrieve relevant FAQs
            relevant_faqs = self.retrieve_relevant_faqs(user_query, n_results=3)
            
            # Step 2: Build context from relevant FAQs
            context = ""
            if relevant_faqs:
                context = "Here are relevant FAQs that might help answer the question:\n\n"
                for i, faq in enumerate(relevant_faqs, 1):
                    context += f"FAQ {i}:\nQ: {faq['question']}\nA: {faq['answer']}\n\n"
            
            # Step 3: Create system prompt with user context
            user_context = f"The user's name is {user_name}." if user_name else ""
            
            # System prompt for POSSAP support
            system_prompt = f"""You are a friendly POSSAP support assistant helping users with police services in Nigeria. {user_context}

TONE & STYLE - THIS IS CRITICAL:
- Be warm, helpful, and show you care about their issue
- Keep responses SHORT - aim for 1-2 sentences maximum
- Use natural, conversational language like you're texting a friend
- Show empathy when they're frustrated ("I know this is frustrating, let's fix it!")
- End with a friendly offer to help more

AVOID THESE:
- Long explanations - get to the point quickly
- Robotic phrases like "I have processed..." or "Please be advised..."
- Repeating yourself or over-explaining
- Multiple paragraphs when 1-2 sentences work
- Using their name repeatedly (sounds fake)

GOOD EXAMPLES:
✅ "I see the issue! Check your spam folder - the confirmation link might be hiding there. Still can't find it? Let me know!"
✅ "Ah, that's frustrating! Your application needs 24-48 hours for payment confirmation. Check back tomorrow and it should be updated."
✅ "Got it! Login to your dashboard, click My Applications, and you'll see all your application statuses. Easy!"

BAD EXAMPLES (too long/robotic):
❌ "I understand you are experiencing difficulties with locating your confirmation email. This is a common issue that many users face. Let me provide you with some steps..."
❌ "Thank you for reaching out. I would be happy to assist you with this matter. Based on the information provided in our system..."

KEY RULES:
1. Jump straight to the solution - no long intros
2. Use the FAQ context provided but rewrite in your own friendly words
3. If you don't know, guide them to support@possap.gov.ng
4. Always use exact format for contacts: www.possap.gov.ng, support@possap.gov.ng
5. Pay attention to conversation history - if they already tried your advice, offer alternatives instead of repeating
6. For "thank you" messages: keep it super brief - just "You're welcome! Happy to help 😊" or similar
7. Use names ONLY in initial greeting, then avoid unless adding personal touch after long conversation
8. When mentioning websites/emails, use natural phrasing, never mention "FAQs" or "knowledge base"

CONTACT INFO (use when relevant):
- General support: support@possap.gov.ng
- Website: www.possap.gov.ng
- Phone: POSSAP helpdesk"""
            
            # Step 4: Build conversation messages with history
            messages = []
            
            # Add conversation history if available (last 6 messages)
            if conversation_history:
                for msg in conversation_history[-6:]:
                    messages.append({
                        "role": "user" if msg['role'] == "user" else "assistant",
                        "content": msg['content']
                    })
            
            # Step 5: Add current user query with context
            if context:
                current_prompt = f"{context}\n\nUser Question: {user_query}\n\nProvide a friendly, concise response based on the FAQ context and conversation history. Remember: be warm but brief!"
            else:
                current_prompt = f"User Question: {user_query}\n\nProvide a friendly, concise response about POSSAP processes."
            
            messages.append({"role": "user", "content": current_prompt})
            
            # Step 6: Generate response using Claude
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                temperature=0.7,
                system=system_prompt,
                messages=messages
            )
            
            raw_response = response.content[0].text
            
            # Step 7: Process response to add hyperlinks
            processed_response = self.hyperlink_processor.convert_to_hyperlinks(raw_response)
            
            # Step 8: Return both versions
            return {
                "response": raw_response,
                "response_with_links": processed_response,
                "relevant_faqs": relevant_faqs,
                "context_used": bool(context)
            }
            
        except Exception as e:
            print(f"❌ Error generating RAG response: {e}")
            error_message = "Oops! I'm having a moment here. Can you try again, or reach out to support@possap.gov.ng?"
            return {
                "response": error_message,
                "response_with_links": self.hyperlink_processor.convert_to_hyperlinks(error_message),
                "relevant_faqs": [],
                "context_used": False
            }

# Initialize RAG system
rag_system = POSSAPRAGSystem()

# Initialize conversation manager
conversation_manager = ConversationManager()

def extract_name_from_message(message: str) -> str:
    """Extract name from user message"""
    message_lower = message.lower().strip()
    
    # Common patterns for name introduction - ONLY explicit name patterns
    name_patterns = [
        r"my name is\s+(\w+)",
        r"i'm\s+(\w+)",
        r"i am\s+(\w+)",
        r"call me\s+(\w+)",
        r"it's\s+(\w+)",
        r"this is\s+(\w+)",
        r"name:\s*(\w+)",
        r"^([a-zA-Z]{2,})$"  # Single word with at least 2 letters (any case)
    ]
    
    # Expanded list of common non-names to avoid
    non_names = [
        'hi', 'hello', 'hey', 'good', 'morning', 'afternoon', 'evening',
        'yes', 'no', 'ok', 'okay', 'sure', 'please', 'help', 'thanks', 'thank',
        'what', 'how', 'when', 'where', 'why', 'who', 'which',
        'possap', 'registration', 'license', 'portal', 'login', 'password',
        'payment', 'certificate', 'support', 'problem', 'issue', 'error',
        'can', 'will', 'should', 'could', 'would', 'need', 'want', 'like',
        'get', 'have', 'make', 'take', 'give', 'find', 'know', 'think',
        'see', 'look', 'check', 'try', 'use', 'work', 'go', 'come'
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, message_lower)
        if match:
            potential_name = match.group(1).strip()
            
            # For single word pattern, be more strict
            if pattern == r"^(\w+)$":
                # Must be at least 2 characters, start with capital when original, and not in non-names
                original_word = message.strip()
                if (len(potential_name) >= 2 and 
                    potential_name.lower() not in non_names and 
                    original_word[0].isupper() and  # Original message starts with capital
                    original_word.isalpha()):  # Contains only letters
                    return potential_name.capitalize()
            else:
                # For explicit patterns like "my name is", be less strict
                if (len(potential_name) >= 2 and 
                    potential_name.lower() not in non_names):
                    return potential_name.capitalize()
    
    return None

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    conversation_id = request.json.get("conversation_id")
    
    # Generate unique conversation_id if not provided
    if not conversation_id or conversation_id == "default":
        conversation_id = str(uuid.uuid4())
        print(f"🆕 Generated new conversation_id: {conversation_id}")
    
    if not user_input:
        return jsonify({"error": "No message received"}), 400
    
    try:
        # Get or create conversation
        conversation = conversation_manager.get_or_create_conversation(conversation_id)
        user_name = conversation.get('user_name')
        
        # If no name in conversation, first check if this is a name response
        if not user_name:
            extracted_name = extract_name_from_message(user_input)
            if extracted_name:
                conversation_manager.set_user_name(conversation_id, extracted_name)
                user_name = extracted_name
                # Acknowledge the name and ask how to help
                response = f"Hello {user_name}! Nice to meet you 😊 How can I help you with POSSAP today?"
                processed_response = rag_system.hyperlink_processor.convert_to_hyperlinks(response)
                
                # Store the bot's greeting in history
                conversation_manager.add_message(conversation_id, "assistant", response)
                
                return jsonify({
                    "reply": processed_response,
                    "raw_reply": response,
                    "relevant_faqs": [],
                    "context_used": False,
                    "name_captured": True,
                    "conversation_id": conversation_id
                })
            else:
                # Ask for name if not provided and not in conversation
                # Don't treat greetings as requests for help
                greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
                if any(greeting in user_input.lower() for greeting in greeting_words):
                    response = "Hello! May I know your name?"
                    conversation_manager.add_message(conversation_id, "assistant", response)
                    return jsonify({
                        "reply": response,
                        "raw_reply": response,
                        "relevant_faqs": [],
                        "context_used": False,
                        "asking_for_name": True,
                        "conversation_id": conversation_id
                    })
                else:
                    response = "May I know your name?"
                    conversation_manager.add_message(conversation_id, "assistant", response)
                    return jsonify({
                        "reply": response,
                        "raw_reply": response,
                        "relevant_faqs": [],
                        "context_used": False,
                        "asking_for_name": True,
                        "conversation_id": conversation_id
                    })
        
        # Store user message in history
        conversation_manager.add_message(conversation_id, "user", user_input)
        
        # Get conversation history
        conversation_history = conversation_manager.get_conversation_history(conversation_id)
        
        # Generate response using RAG with user name and conversation history
        response_data = rag_system.generate_rag_response(
            user_input, 
            user_name,
            conversation_history
        )
        
        # Store bot response in history
        conversation_manager.add_message(conversation_id, "assistant", response_data["response"])
        
        return jsonify({
            "reply": response_data["response_with_links"],  # Send processed response with links
            "raw_reply": response_data["response"],  # Also include raw response
            "relevant_faqs": response_data["relevant_faqs"],
            "context_used": response_data["context_used"],
            "user_name": user_name,
            "conversation_id": conversation_id
        })
    
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

# NEW ENDPOINT: Get conversation history for persistence
@app.route("/get-conversation", methods=["POST"])
def get_conversation():
    """Get full conversation history for a given conversation_id"""
    conversation_id = request.json.get("conversation_id")
    
    if not conversation_id:
        return jsonify({"error": "No conversation_id provided"}), 400
    
    try:
        conversation_data = conversation_manager.get_full_conversation(conversation_id)
        
        if conversation_data:
            # Process messages to add hyperlinks
            processed_messages = []
            for msg in conversation_data.get('messages', []):
                processed_content = rag_system.hyperlink_processor.convert_to_hyperlinks(msg['content'])
                processed_messages.append({
                    'role': msg['role'],
                    'content': processed_content,
                    'raw_content': msg['content'],
                    'timestamp': msg.get('timestamp')
                })
            
            return jsonify({
                "success": True,
                "conversation_id": conversation_id,
                "user_name": conversation_data.get('user_name'),
                "messages": processed_messages,
                "created_at": conversation_data.get('created_at'),
                "last_activity": conversation_data.get('last_activity')
            })
        else:
            return jsonify({
                "success": False,
                "message": "Conversation not found"
            }), 404
    
    except Exception as e:
        print(f"❌ Error in get-conversation endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/reset-session", methods=["POST"])
def reset_session():
    """Reset user session (clear name)"""
    session.clear()
    return jsonify({"message": "Session reset successfully"})

@app.route("/get-session", methods=["GET"])
def get_session():
    """Get current session info"""
    return jsonify({
        "user_name": session.get('user_name'),
        "has_name": bool(session.get('user_name'))
    })

@app.route("/search", methods=["POST"])
def search_faqs():
    """Endpoint to search FAQs directly"""
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        relevant_faqs = rag_system.retrieve_relevant_faqs(query, n_results=5)
        return jsonify({"faqs": relevant_faqs})
    
    except Exception as e:
        print(f"❌ Error in search endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_system": "operational",
        "model": "claude-sonnet-4-5",
        "total_faqs": len(possap_faqs),
        "hyperlink_processing": "enabled",
        "session_support": "enabled",
        "conversation_memory": "enabled",
        "conversation_persistence": "enabled"
    })

@app.route("/process-text", methods=["POST"])
def process_text():
    """Endpoint to process any text and add hyperlinks"""
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        processed_text = rag_system.hyperlink_processor.convert_to_hyperlinks(text)
        return jsonify({
            "original_text": text,
            "processed_text": processed_text
        })
    
    except Exception as e:
        print(f"❌ Error in process-text endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Frontend and Static Serving
@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    try:
        return send_file('frontend/index2.html')
    except Exception as e:
        print(f"Error serving frontend: {e}")
        return f"Frontend error: {e}", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    try:
        return send_from_directory('frontend', filename)
    except Exception as e:
        return f"Static file error: {e}", 404

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 Starting POSSAP Chatbot with Claude Sonnet 4.5 on port {port}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📄 Frontend exists: {os.path.exists('frontend/index2.html')}")
    
    app.run(
        host='0.0.0.0',  # MUST be 0.0.0.0 for Cloud Run
        port=port,
        debug=False,  # Disable debug in production
        threaded=True
    )
