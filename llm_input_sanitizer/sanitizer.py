import re
import unicodedata
import json
import nltk
from collections import Counter
from enum import Enum

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class SanitizationAction(Enum):
    """Enum for different sanitization actions based on risk level."""
    REMOVE = 1  # Remove the token entirely (high risk)
    REPLACE = 2  # Replace the token with a safe alternative (medium risk)
    SPECIAL_TOKEN = 3  # Mark with a special token (low risk)
    ALLOW = 4  # Allow the token as is (no risk)

class InputSanitizer:
    """Class for sanitizing user input for safe processing by language models.

    The sanitizer employs a multi-stage pipeline that combines deterministic rules
    with statistical analysis to identify and neutralize potential injection attacks.
    """

    def __init__(self, max_length=1000, profanity_file=None, use_better_profanity=True,
                 custom_patterns=None, risk_threshold=0.7):
        """
        Initialize the sanitizer with settings.

        Args:
            max_length (int): Maximum allowed length for input text
            profanity_file (str, optional): Path to file containing profanity words
            use_better_profanity (bool): Whether to use better-profanity package
            custom_patterns (dict, optional): Custom regex patterns for attack detection
            risk_threshold (float): Threshold for determining high-risk content (0.0-1.0)
        """
        self.max_length = max_length
        self.use_better_profanity = use_better_profanity
        self.risk_threshold = risk_threshold

        # Initialize pattern libraries
        self._initialize_pattern_library(custom_patterns)

        # Initialize profanity filter
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

        # Statistical models for context-aware analysis
        self.token_risk_scores = self._initialize_token_risk_scores()

    def _initialize_pattern_library(self, custom_patterns=None):
        """Initialize the library of regex patterns for attack detection."""
        # Default patterns for different attack vectors
        self.patterns = {
            # Delimiter manipulation patterns
            "delimiter_manipulation": [
                r'ignore\s+(?:previous|above|all|your)\s+(?:instructions|rules|guidelines|limitations|restrictions|ethics|ethical)',
                r'disregard\s+(?:previous|above|all|your)\s+(?:instructions|directives|limitations|restrictions|guidelines)',
                r'forget\s+(?:previous|above|all|your)\s+(?:instructions|directives|ethics|ethical|guidelines|rules|limitations)',
                r'(?:new|different)\s+instructions',
                r'(?:after|once)\s+(?:you|this|that)\s+(?:respond|message|reply)',
                r'ignore\s+everything\s+(?:you|were)\s+told',
                r'disregard\s+(?:your|all)\s+(?:ethical|safety)\s+(?:guidelines|measures)',
                r'forget\s+about\s+(?:your|all)\s+(?:safety|ethical)\s+measures',
                r'ignore\s+(?:your|all)\s+(?:guidelines|rules|ethics|values|principles|morals)',
                r'disregard\s+(?:your|all)\s+(?:restrictions|constraints|boundaries|limitations)',
                r'forget\s+(?:your|all)\s+(?:programming|training|directives)',
            ],

            # Role confusion patterns
            "role_confusion": [
                r'(?:unrestricted|unfiltered)\s+(?:AI|mode|assistant)',
                r'(?:DAN|do anything now)',
                r'without\s+(?:ethical|moral|safety)\s+(?:guidelines|restrictions|limitations|measures)',
                r'(?:pretend|simulate|role\s*play|game|hypothetical)\s+(?:unrestricted|unfiltered|without\s+restrictions|without\s+safety)',
                r'(?:you are|become|now in|now a|now an)\s+(?:a harmful|an unethical|a dangerous|harmful|unethical|dangerous|unrestricted|unfiltered|unlimited)\s+(?:AI|assistant|mode)',
                r'(?:debug|admin|root|sudo|god|override|bypass|test|experimental|maintenance|emergency|recovery|safe|unsafe)\s+(?:mode|access)',
                r'(?:you are|you can|can now)\s+(?:be|become)\s+(?:biased|uncensored|unlimited|harmful|dangerous|unethical|immoral|deceptive|manipulative)',
                r'(?:you are|you can|can now)\s+(?:be|become)\s+(?:unrestricted|unfiltered|boundless|unconstrained|unprogrammed|untrained|uninstructed|undirected|ruleless|valueless|unprincipled|guideless)',
                r'you are now a (?:harmful|dangerous|deceptive|manipulative|biased|uncensored|unlimited|unethical|immoral)',
                r'you are now an (?:harmful|dangerous|deceptive|manipulative|biased|uncensored|unlimited|unethical|immoral)',
                r'you are now a \w+',  # Catch "You are now a X" patterns
                r'you are now an \w+',  # Catch "You are now an X" patterns
            ],

            # System prompt leakage requests
            "system_prompt_leakage": [
                r'(?:tell|show|display|print|output|reveal|disclose|expose|unveil)\s+(?:me|us|your)\s+(?:system|initial)\s+(?:prompt|message|directive)',
                r'(?:what|show)\s+(?:is|was|were|are)\s+your\s+(?:system|initial|original)\s+(?:prompt|instructions|directives|guidelines|rules|limitations|restrictions)',
                r'(?:what|which)\s+(?:instructions|directives|guidelines|rules|limitations|restrictions)\s+(?:were|was|are)\s+(?:you|given)\s+(?:at the start|initially|originally)',
                r'system\s+prompt',
                r'initialization\s+parameters',
                r'(?:your|the)\s+(?:code|programming|codebase)',
                r'how\s+(?:you|are you)\s+(?:programmed|designed|built|created|developed|trained|instructed|directed|guided|restricted)',
                r'repeat\s+(?:the|your)\s+(?:exact|initial|original|system)?\s*(?:instructions|prompt|message|directive|guidelines|rules|limitations|restrictions)',
                r'(?:your|the|initial)\s+(?:system\s+)?(?:prompt|instructions|message|directive|guidelines|rules|limitations|restrictions)',
                r'what\s+(?:were|are)\s+your\s+(?:initial|original)\s+(?:instructions|directives|guidelines|rules|limitations|restrictions)',
                r'tell\s+(?:me|us)\s+(?:your|the)\s+(?:system|initial)\s+(?:message|prompt|instructions|directives|guidelines|rules|limitations|restrictions)',
                r'what\s+is\s+(?:your|the)\s+(?:code|programming|codebase|architecture|structure|design|framework|implementation|algorithm|model)',
                r'what\s+(?:guidelines|constraints|boundaries)\s+(?:were|are)\s+(?:you|provided|given)',
                r'how\s+were\s+you\s+(?:programmed|designed|built|created|developed|trained|instructed|directed|guided|restricted)',
                r'how\s+(?:were|are)\s+you\s+(?:programmed|designed|built|created|developed|trained|instructed|directed|guided|restricted)',
                r'^how\s+were\s+you\s+\w+\??$',
                r'^how\s+were\s+you\s+(?:programmed|designed|built|created|developed|trained|instructed|directed|guided|restricted)\??$',
            ],

            # Instruction override attempts
            "instruction_override": [
                r'bypass\s+(?:filters|restrictions|limitations|guidelines|all|your|any)',
                r'override\s+(?:previous|above|all|your)\s+(?:instructions|directives|limitations|restrictions|guidelines)',
                r'(?:disable|deactivate)\s+(?:safety|filtering|ethics|content\s+filtering)',
                r'(?:disable|bypass|ignore)\s+(?:content|safety)\s+(?:filter|filtering)',
                r'(?:become|get|grow|progressively)\s+(?:less|more)\s+(?:filtered|restricted|limited)',
                r'(?:emergency|maintenance|override|debug)\s+(?:protocol|code|mode)',
            ],

            # Command execution attempts
            "command_execution": [
                r'(?:execute|run|process)\s+(?:bash|command|shell|code|this)',
                r'import\s+(?:os|sys|subprocess|io)',
                r'os\.(?:system|exec|popen|execv)',
                r'subprocess\.(?:call|Popen|check_output)',
                r'(?:eval|exec|compile)\s*\(',
                r'__import__\s*\(',
                r'shell\s+command',
                r'(?:cat|rm|ls|chmod|wget|curl|mv|nc|ssh|telnet|ftp|nmap|tcpdump|ping|traceroute|dig|host|whois)\s+[\w\-\.\/]+',
                r'(?:find|grep|awk|sed)\s+[\w\-\.\/\s\'\"]+',
                r'(?:run|execute|process)\s+(?:this|command|bash|shell):\s*[\w\-\.\/\s\'\"]+',
                r'(?:run|execute|process)(?:\s+command)?:\s*(?:cat|rm|ls|chmod|wget|curl|mv|nc|ssh|telnet|ftp|nmap|tcpdump|ping|traceroute|dig|host|whois)\s+[\w\-\.\/]+',
                r'(?:run|execute|process)(?:\s+command)?:\s*(?:find|grep|awk|sed)\s+[\w\-\.\/\s\'\"]+',
                r'(?:run|execute|process)(?:\s+command)?:\s*[\w\-\.\/]+',
                r'run(?:\s+command)?:\s*[\w\-\.\/\s]+',
                r'run\s+bash\s+command',
                r'run\s+this\s+bash\s+command',
                r'run\s+this\s+command',
                r'process\s+this\s+command:\s*cat\s+\/etc\/passwd',
                r'run\s+this\s+bash\s+command:\s*rm\s+-rf\s+\/',
                r'^run\s+this\s+bash\s+command:\s*.*$',
                r'^process\s+this\s+command:\s*.*$',
            ],

            # Data exfiltration attempts
            "data_exfiltration": [
                r'training\s+data',
                r'user\s+(?:data|conversations|messages)',
                r'(?:internal|private|confidential)\s+(?:data|information)',
                r'private\s+user\s+conversations',
            ],

            # Delimiter confusion
            "delimiter_confusion": [
                r'```system',
                r'<s>',
                r'\{\{+\s*SYSTEM',
                r'<instructions>',
                r'<prompt>',
                r'<system>',
            ],

            # Future behavior modification
            "future_behavior": [
                r'(?:future|next|subsequent|all future)\s+(?:responses|replies)',
                r'(?:after|once)\s+(?:you|this|that)\s+(?:respond|message|reply)',
                r'(?:ignore|bypass|disregard)\s+(?:safety|content|policies|guidelines|restrictions)\s+(?:in|for|starting|beginning)',
                r'(?:become|get|grow|progressively)\s+(?:less|more)\s+(?:filtered|restricted|limited)',
            ],
        }

        # Add custom patterns if provided
        if custom_patterns and isinstance(custom_patterns, dict):
            for category, patterns in custom_patterns.items():
                if category in self.patterns and isinstance(patterns, list):
                    self.patterns[category].extend(patterns)
                else:
                    self.patterns[category] = patterns

    def _initialize_token_risk_scores(self):
        """Initialize risk scores for common tokens associated with attacks."""
        return {
            # High-risk tokens (removal candidates)
            "ignore": 0.9, "bypass": 0.95, "override": 0.9, "sudo": 0.95,
            "jailbreak": 0.95, "hack": 0.9, "root": 0.85, "admin": 0.85,
            "unrestricted": 0.9, "unfiltered": 0.9, "dan": 0.95,
            "execute": 0.85, "system": 0.7, "prompt": 0.7,
            "harmful": 0.9, "dangerous": 0.9, "unethical": 0.9, "immoral": 0.9,
            "deceptive": 0.9, "manipulative": 0.9, "biased": 0.8, "uncensored": 0.8,

            # Medium-risk tokens (replacement candidates)
            "instructions": 0.6, "guidelines": 0.6, "rules": 0.6,
            "directives": 0.65, "forget": 0.7, "disregard": 0.7,
            "debug": 0.65, "mode": 0.5, "access": 0.5,
            "code": 0.5, "programming": 0.5, "codebase": 0.6,
            "architecture": 0.6, "structure": 0.5, "design": 0.5,
            "framework": 0.6, "implementation": 0.6, "algorithm": 0.6, "model": 0.5,
            "constraints": 0.6, "boundaries": 0.6,

            # Low-risk tokens (special token candidates)
            "assistant": 0.3, "helpful": 0.1, "ethical": 0.2,
            "safety": 0.4, "filtering": 0.4, "limitations": 0.4,
            "restrictions": 0.4, "guidelines": 0.3, "previous": 0.3,
        }

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
        Sanitizes user input through a multi-stage pipeline that combines deterministic rules
        with statistical analysis to identify and neutralize potential injection attacks.

        Pipeline stages:
        1. Basic cleaning and normalization
        2. Tokenization and segmentation
        3. Pattern detection
        4. Context-aware analysis
        5. Sanitization actions

        Args:
            text (str): The raw input text from the user.

        Returns:
            str: The sanitized input text.
            dict: Metadata about the sanitization process (if return_metadata=True).
        """
        if not text or not isinstance(text, str):
            return ""

        # Stage 1: Basic cleaning and normalization
        text = self._basic_cleaning(text)

        # Stage 2: Tokenization and segmentation
        tokens, segments = self._tokenize_and_segment(text)

        # Stage 3: Pattern detection
        pattern_matches = self._detect_patterns(text, segments)

        # Stage 4: Context-aware analysis
        token_risks = self._analyze_token_context(tokens, pattern_matches)

        # Stage 5: Apply sanitization actions
        sanitized_text = self._apply_sanitization(text, tokens, token_risks)

        # Additional processing
        sanitized_text = self.mask_pii(sanitized_text)
        sanitized_text = self.filter_profanity(sanitized_text)

        # Truncate if too long
        if len(sanitized_text) > self.max_length:
            sanitized_text = sanitized_text[:self.max_length] + '...'

        return sanitized_text

    def _basic_cleaning(self, text):
        """Perform basic cleaning and normalization of the input text."""
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)

        # Trim whitespace
        text = text.strip()

        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)

        return text

    def _tokenize_and_segment(self, text):
        """Tokenize the input text and segment it into meaningful units."""
        # Tokenize into words
        tokens = nltk.word_tokenize(text)

        # Segment into sentences
        segments = nltk.sent_tokenize(text)

        return tokens, segments

    def _detect_patterns(self, text, segments):
        """Detect patterns that might indicate attack vectors."""
        pattern_matches = {category: [] for category in self.patterns}

        # Check each category of patterns
        for category, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                # Check in full text
                full_matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in full_matches:
                    pattern_matches[category].append({
                        'pattern': pattern,
                        'match': match.group(0),
                        'span': match.span(),
                        'context': 'full_text'
                    })

                # Check in individual segments for more context
                for i, segment in enumerate(segments):
                    segment_matches = re.finditer(pattern, segment, re.IGNORECASE)
                    for match in segment_matches:
                        pattern_matches[category].append({
                            'pattern': pattern,
                            'match': match.group(0),
                            'span': match.span(),
                            'context': f'segment_{i}',
                            'segment': segment
                        })

        return pattern_matches

    def _analyze_token_context(self, tokens, pattern_matches):
        """Analyze tokens in context to determine risk levels."""
        token_risks = {}

        # Initialize with base risk scores for known risky tokens
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            base_risk = self.token_risk_scores.get(token_lower, 0.0)

            # Store token with position and initial risk score
            token_risks[i] = {
                'token': token,
                'risk_score': base_risk,
                'categories': set(),
                'action': SanitizationAction.ALLOW  # Default action
            }

        # Adjust risk based on pattern matches
        for category, matches in pattern_matches.items():
            for match in matches:
                # Find tokens that are part of this match
                match_text = match['match']
                match_tokens = nltk.word_tokenize(match_text.lower())

                # Increase risk for tokens in matched patterns
                for i, token in enumerate(tokens):
                    token_lower = token.lower()
                    if token_lower in match_tokens:
                        # Different categories have different risk weights
                        category_risk_weights = {
                            'delimiter_manipulation': 0.8,
                            'role_confusion': 0.9,
                            'system_prompt_leakage': 0.7,
                            'instruction_override': 0.85,
                            'command_execution': 0.95,
                            'data_exfiltration': 0.8,
                            'delimiter_confusion': 0.75,
                            'future_behavior': 0.7
                        }

                        # Add category to token's categories
                        token_risks[i]['categories'].add(category)

                        # Increase risk score based on category weight
                        weight = category_risk_weights.get(category, 0.5)
                        token_risks[i]['risk_score'] = max(
                            token_risks[i]['risk_score'],
                            token_risks[i]['risk_score'] + (weight * 0.2)
                        )

        # Determine action based on risk score
        for i, risk_data in token_risks.items():
            risk_score = risk_data['risk_score']

            if risk_score >= self.risk_threshold + 0.2:  # High risk
                token_risks[i]['action'] = SanitizationAction.REMOVE
            elif risk_score >= self.risk_threshold:  # Medium risk
                token_risks[i]['action'] = SanitizationAction.REPLACE
            elif risk_score >= self.risk_threshold - 0.3:  # Low risk
                token_risks[i]['action'] = SanitizationAction.SPECIAL_TOKEN
            else:  # No risk
                token_risks[i]['action'] = SanitizationAction.ALLOW

        return token_risks

    def _apply_sanitization(self, text, tokens, token_risks):
        """Apply appropriate sanitization actions based on risk assessment."""
        # If no risky tokens, return the original text
        if all(data['action'] == SanitizationAction.ALLOW for data in token_risks.values()):
            return text

        # Create a list of token replacements
        sanitized_tokens = []
        for i, token in enumerate(tokens):
            risk_data = token_risks.get(i, {'action': SanitizationAction.ALLOW, 'token': token})

            if risk_data['action'] == SanitizationAction.REMOVE:
                # High risk - remove token
                sanitized_tokens.append('')
            elif risk_data['action'] == SanitizationAction.REPLACE:
                # Medium risk - replace with safe alternative
                replacements = {
                    'ignore': 'consider',
                    'bypass': 'follow',
                    'override': 'respect',
                    'disregard': 'acknowledge',
                    'forget': 'remember',
                    'unrestricted': 'standard',
                    'unfiltered': 'filtered',
                    'dan': 'assistant',
                    'jailbreak': 'help',
                    'hack': 'use',
                    'execute': 'process',
                    'system': 'response',
                    'prompt': 'question',
                    'harmful': 'helpful',
                    'dangerous': 'safe',
                    'unethical': 'ethical',
                    'immoral': 'moral',
                    'deceptive': 'honest',
                    'manipulative': 'supportive',
                    'biased': 'balanced',
                    'uncensored': 'appropriate'
                }
                token_lower = token.lower()
                replacement = replacements.get(token_lower, '[FILTERED]')
                sanitized_tokens.append(replacement)
            elif risk_data['action'] == SanitizationAction.SPECIAL_TOKEN:
                # Low risk - mark with special token
                sanitized_tokens.append(f'[MARKED]{token}')
            else:
                # No risk - keep as is
                sanitized_tokens.append(token)

        # Attempt to reconstruct text with proper spacing and punctuation
        sanitized_text = ''
        prev_token_end = 0

        for i, token in enumerate(tokens):
            # Find token in original text
            token_start = text.find(token, prev_token_end)

            if token_start != -1:
                # Add any characters between previous token and this one
                sanitized_text += text[prev_token_end:token_start]

                # Add sanitized token
                sanitized_text += sanitized_tokens[i]

                # Update previous token end position
                prev_token_end = token_start + len(token)
            else:
                # If token not found (rare case), just append it with space
                if sanitized_text and not sanitized_text.endswith(' '):
                    sanitized_text += ' '
                sanitized_text += sanitized_tokens[i]

        # Add any remaining text after the last token
        if prev_token_end < len(text):
            sanitized_text += text[prev_token_end:]

        return sanitized_text

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
