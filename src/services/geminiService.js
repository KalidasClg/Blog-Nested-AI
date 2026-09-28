const { GoogleGenerativeAI } = require('@google/generative-ai');

const DEFAULT_MODEL = 'gemini-3.8-flash';
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
const UNAVAILABLE_STATUS = new Set([404]);
const DEFAULT_MAX_ATTEMPTS = 3;
const BASE_DELAY_MS = 500;
const MAX_DELAY_MS = 8000;

const stripPrefix = (value) => String(value).trim().replace(/^models\//, '');

const resolveModelCandidates = () => {
  const configured = [
    process.env.GEMINI_MODEL,
    ...String(process.env.GEMINI_FALLBACK_MODELS || '').split(','),
  ]
    .map(stripPrefix)
    .filter(Boolean);

  const candidates = configured.length ? configured : [DEFAULT_MODEL];

  return [...new Set([...candidates, DEFAULT_MODEL])];
};

const resolveMaxAttempts = () => {
  const parsed = parseInt(process.env.GEMINI_MAX_ATTEMPTS, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_ATTEMPTS;
};

const getStatus = (error) => {
  const raw = error && (error.status !== undefined ? error.status : error.code);
  const numeric = typeof raw === 'string' ? parseInt(raw, 10) : raw;
  return Number.isFinite(numeric) ? numeric : undefined;
};

const isRetryable = (error) => RETRYABLE_STATUS.has(getStatus(error));

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const backoffDelay = (attempt) => {
  const exponential = Math.min(BASE_DELAY_MS * 2 ** attempt, MAX_DELAY_MS);
  return exponential / 2 + Math.random() * (exponential / 2);
};

const generateOnce = async (genAI, modelName, prompt) => {
  const model = genAI.getGenerativeModel({ model: modelName });
  const result = await model.generateContent(prompt);
  const response = await result.response;
  const text = response.text();

  if (!text) {
    throw new Error('Invalid Gemini response');
  }

  return text;
};

const callGemini = async (prompt) => {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    const error = new Error('Gemini API key is not configured');
    error.status = 500;
    throw error;
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const candidates = resolveModelCandidates();
  const maxAttempts = resolveMaxAttempts();
  let lastError;
  let lastUnavailable = false;

  for (const modelName of candidates) {
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      try {
        return await generateOnce(genAI, modelName, prompt);
      } catch (error) {
        lastError = error;
        const status = getStatus(error);

        if (UNAVAILABLE_STATUS.has(status)) {
          lastUnavailable = true;
          console.warn(
            `Gemini model "${modelName}" is unavailable (${status}) - falling back to next model`
          );
          break;
        }

        lastUnavailable = false;

        if (!isRetryable(error)) {
          console.error(`Gemini API Error (${modelName}):`, error.message);
          throw error;
        }

        const willRetry = attempt < maxAttempts - 1;
        console.warn(
          `Gemini ${status} on ${modelName} (attempt ${attempt + 1}/${maxAttempts})${willRetry ? ' - retrying' : ''}`
        );

        if (willRetry) {
          await sleep(backoffDelay(attempt));
        }
      }
    }
  }

  console.error('Gemini API Error: all models exhausted -', lastError && lastError.message);

  if (lastUnavailable) {
    const noModel = new Error(
      'The configured AI model is not available for this API key. Check GEMINI_MODEL in your .env file.'
    );
    noModel.status = 502;
    throw noModel;
  }

  const overloaded = new Error(
    'The AI service is temporarily overloaded. Please retry in a few moments.'
  );
  overloaded.status = 503;
  overloaded.retryable = true;
  throw overloaded;
};

module.exports = {
  callGemini,
};
