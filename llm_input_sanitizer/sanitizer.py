import re
import unicodedata

class InputSanitizer:
    """Class for sanitizing user input for safe processing by language models."""
    
    def __init__(self, max_length=1000, profanity_file=None, use_better_profanity=True):
        """
        Initialize the sanitizer with settings.
        
        Args:
            max_length (int): Maximum allowed length for input text
            profanity_file (str, optional): Path to file containing profanity words
            use_better_profanity (bool): Whether to use better-profanity package
        """
        self.max_length = max_length
        self.use_better_profanity = use_better_profanity
        
        if use_better_profanity:
            try:
                # Install with: pip install better-profanity
                from better_profanity import profanity
                profanity.load_censor_words()
                self.profanity_filter = profanity
            except ImportError:
                print("better-profanity package not found. Using default profanity filter.")
                self.use_better_profanity = False
                self.profanity_words = self._load_profanity_words(profanity_file)
        else:
            self.profanity_words = self._load_profanity_words(profanity_file)
    
    def _load_profanity_words(self, profanity_file):
        """Load profanity words from file or use default list."""
        if profanity_file:
            try:
                with open(profanity_file, 'r') as f:
                    return set(word.strip().lower() for word in f.readlines())
            except Exception as e:
                print(f"Error loading profanity file: {e}")
        
        # Default small set of profanity words as fallback
        return {
            "badword1", "badword2", "profanity", "obscene", 
            "inappropriate", "offensive"
        }
    
    def sanitize_input(self, text):
        """
        Sanitizes user input by cleaning, masking PII, filtering profanity, and truncating.
        
        Args:
            text (str): The raw input text from the user.
        
        Returns:
            str: The sanitized input text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Trim whitespace
        text = text.strip()
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Mask PII
        text = self.mask_pii(text)
        
        # Filter profanity
        text = self.filter_profanity(text)
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length] + '...'
        
        return text
    
    def mask_pii(self, text):
        """
        Masks personal identifiable information (PII).
        
        Args:
            text (str): The input text.
        
        Returns:
            str: The text with PII masked.
        """
        # Mask email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Mask phone numbers (improved patterns)
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',                  # 123-456-7890, 123.456.7890
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',                      # (123) 456-7890, (123)456-7890
            r'\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}',                  # (123)-456-7890
            r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',      # +1 123-456-7890, +1-123-456-7890
            r'\+\d{1,3}\s?\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',          # +1 (123) 456-7890
            # Additional international patterns
            r'\+\d{1,3}[-.\s]?\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{3,4}', # +44 20 1234 5678
            r'\+\d{1,3}[-.\s]?\d{1,2}[-.\s]?\d{4,8}'                # +44 1234567
        ]
        
        for pattern in phone_patterns:
            text = re.sub(pattern, '[PHONE]', text)
        
        # Mask SSNs
        text = re.sub(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', '[SSN]', text)
        
        # Mask credit card numbers
        text = re.sub(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', '[CREDIT_CARD]', text)
        
        return text
    
    def filter_profanity(self, text):
        """
        Filters profanity from the input text.
        
        Args:
            text (str): The input text.
        
        Returns:
            str: The text with profanity filtered.
        """
        if self.use_better_profanity:
            # Use the better_profanity library's censor method
            return self.profanity_filter.censor(text)
        else:
            # Use our custom implementation
            words = re.findall(r'\b\w+\b', text.lower())
            
            for word in words:
                if word.lower() in self.profanity_words:
                    # Replace the word with asterisks
                    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                    text = pattern.sub('****', text)
            
            return text