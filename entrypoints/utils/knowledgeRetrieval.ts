/**
 * Echo — Personal Knowledge Retrieval Engine V1
 * Deterministic keyword-based retrieval of grounded personal facts from IndexedDB.
 */

import {
  db,
  getPersonalKnowledge,
  getKnowledgeEvidence,
  type EchoDatabase,
  type PersonalKnowledgeItem,
  type KnowledgeEvidence,
} from './database';

export const COMMON_STOP_WORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
  'any', 'are', 'aren', 'arent', 'as', 'at', 'be', 'because', 'been', 'before',
  'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cant', 'cannot',
  'could', 'couldnt', 'did', 'didn', 'didnt', 'do', 'does', 'doesn', 'doesnt',
  'doing', 'don', 'dont', 'down', 'during', 'each', 'few', 'for', 'from',
  'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he',
  'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if',
  'in', 'into', 'is', 'isn', 'isnt', 'it', 'its', 'itself', 'just', 'me', 'more',
  'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once',
  'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
  'she', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that', 'the',
  'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
  'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
  'was', 'wasnt', 'we', 'were', 'werent', 'what', 'when', 'where', 'which',
  'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'wont', 'would',
  'wouldnt', 'you', 'your', 'yours', 'yourself', 'yourselves'
]);

/**
 * Procedural intent and formatting tokens describing prompt structure or output format,
 * rather than the subject matter or personal knowledge domain.
 * Filtering these prevents generic requests (e.g. "casual social media post about X")
 * from matching unrelated facts that mention "post" or "social".
 */
export const PROMPT_INTENT_STOP_WORDS = new Set([
  'post', 'posts', 'social', 'media', 'casual', 'write', 'writing', 'draft',
  'drafts', 'tweet', 'tweets', 'article', 'articles', 'essay', 'essays',
  'paragraph', 'paragraphs', 'sentence', 'sentences', 'summary', 'summaries',
  'reflection', 'reflections', 'thoughts', 'update', 'updates', 'note', 'notes',
  'message', 'messages', 'share', 'sharing', 'tell', 'talk', 'talking', 'create',
  'content', 'short', 'medium', 'long', 'professional', 'simple',
]);

export interface RetrievedKnowledgeResult {
  items: PersonalKnowledgeItem[];
  formattedStrings: string[];
  keywordsUsed: string[];
  hasRelevantKnowledge: boolean;
  isAutobiographicalWithoutFacts?: boolean;
  safetyNotice?: string;
}

/**
 * Extracts meaningful non-stopword tokens from a given topic or text.
 * Filters both general stop words and prompt-intent formatting terms when substantive keywords exist.
 */
export function extractTopicKeywords(topic: string): string[] {
  if (!topic || typeof topic !== 'string') {
    return [];
  }

  // Normalize and extract words
  const rawTokens = topic
    .toLowerCase()
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 2);

  const meaningfulKeywords: string[] = [];
  const fallbackKeywords: string[] = [];
  const seen = new Set<string>();

  for (const token of rawTokens) {
    if (!COMMON_STOP_WORDS.has(token) && !seen.has(token)) {
      seen.add(token);
      fallbackKeywords.push(token);
      if (!PROMPT_INTENT_STOP_WORDS.has(token)) {
        meaningfulKeywords.push(token);
      }
    }
  }

  // Use substantive keywords if any exist; otherwise gracefully fall back to non-stop tokens
  return meaningfulKeywords.length > 0 ? meaningfulKeywords : fallbackKeywords;
}

// Irregular English verb/noun stem mappings for lightweight retrieval
const IRREGULAR_STEMS: Record<string, string> = {
  built: 'build',
  thought: 'think',
  thoughts: 'think',
  wrote: 'write',
  written: 'write',
  knew: 'know',
  known: 'know',
  knowledge: 'know',
  began: 'begin',
  begun: 'begin',
  ran: 'run',
};

/**
 * Lightweight, deterministic English stemmer for browser execution.
 * Handles common morphological variations:
 * - argue ↔ arguments ↔ arguing ↔ argument
 * - listen ↔ listening ↔ listened
 * - learn ↔ learning ↔ learned
 * - build ↔ building ↔ built
 * - code ↔ coding ↔ codes
 */
export function stemWord(word: string): string {
  if (!word || typeof word !== 'string') return '';
  let w = word.toLowerCase().trim();
  if (w.length <= 3) return w;

  const irregular = IRREGULAR_STEMS[w];
  if (irregular) {
    return irregular;
  }

  // Common morphological suffixes in English
  if (w.endsWith('ments')) w = w.slice(0, -5);
  else if (w.endsWith('ment')) w = w.slice(0, -4);
  else if (w.endsWith('ations')) w = w.slice(0, -6);
  else if (w.endsWith('ation')) w = w.slice(0, -5);
  else if (w.endsWith('tions') || w.endsWith('sions')) w = w.slice(0, -5);
  else if (w.endsWith('tion') || w.endsWith('sion')) w = w.slice(0, -4);
  else if (w.endsWith('ings')) w = w.slice(0, -4);
  else if (w.endsWith('ing') && w.length > 5) {
    w = w.slice(0, -3);
    const last = w.charAt(w.length - 1);
    const prev = w.charAt(w.length - 2);
    if (w.length >= 4 && last === prev && last !== 'l' && last !== 's') {
      w = w.slice(0, -1);
    }
  } else if (w.endsWith('ies') && w.length > 4) {
    w = w.slice(0, -3) + 'y';
  } else if (w.endsWith('ed') && w.length > 4) {
    w = w.slice(0, -2);
    const last = w.charAt(w.length - 1);
    const prev = w.charAt(w.length - 2);
    if (w.length >= 4 && last === prev && last !== 'l' && last !== 's') {
      w = w.slice(0, -1);
    }
  } else if (w.endsWith('ers') && w.length > 5) {
    w = w.slice(0, -3);
  } else if (w.endsWith('er') && w.length > 4) {
    w = w.slice(0, -2);
  } else if (w.endsWith('es') && w.length > 4 && !w.endsWith('ies')) {
    w = w.slice(0, -2);
  } else if (w.endsWith('s') && !w.endsWith('ss') && w.length > 3) {
    w = w.slice(0, -1);
  }

  // Trim trailing silent 'e' for consistent verb/noun unification (e.g. argue -> argu, code -> cod)
  if (w.endsWith('e') && w.length > 3) {
    w = w.slice(0, -1);
  }

  return w;
}

/**
 * Checks whether a subject string represents an author/user meta-placeholder
 * (e.g. 'the user', 'user', 'author') rather than a specific project or domain topic.
 */
export function isAuthorSubject(subject: string): boolean {
  const s = (subject || '').toLowerCase().trim();
  return (
    s === 'the user' ||
    s === 'user' ||
    s === 'author' ||
    s === 'the author' ||
    s === 'i' ||
    s === 'me' ||
    s === 'myself' ||
    s.startsWith('i ') ||
    s.startsWith('my ')
  );
}

function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Tests for whole-word exact matches and morphological stem matches in text tokens.
 */
export function matchKeywordInText(kw: string, kwStem: string, text: string): { exact: boolean; stem: boolean } {
  if (!text) return { exact: false, stem: false };

  // 1. Exact whole-word boundary match
  const kwRegex = new RegExp(`\\b${escapeRegExp(kw)}\\b`, 'i');
  if (kwRegex.test(text)) {
    return { exact: true, stem: false };
  }

  // 2. Stem match across individual tokens
  if (kwStem && kwStem.length >= 3) {
    const tokens = text
      .toLowerCase()
      .replace(/[^\w\s-]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length >= 2);

    for (const t of tokens) {
      if (stemWord(t) === kwStem) {
        return { exact: false, stem: true };
      }
    }
  }

  return { exact: false, stem: false };
}

export function toFirstPersonVerb(verb: string): string {
  const trimmed = (verb || '').trim();
  const v = trimmed.toLowerCase();
  if (v === 'uses' || v.startsWith('uses ')) return 'use' + trimmed.slice(4);
  if (v === 'plans' || v.startsWith('plans ')) return 'plan' + trimmed.slice(5);
  if (v === 'plans to learn') return 'plan to learn';
  if (v === 'wants' || v.startsWith('wants ')) return 'want' + trimmed.slice(5);
  if (v === 'wants to learn') return 'want to learn';
  if (v === 'aims' || v.startsWith('aims ')) return 'aim' + trimmed.slice(4);
  if (v === 'aims to') return 'aim to';
  if (v === 'believes' || v.startsWith('believes ')) return 'believe' + trimmed.slice(8);
  if (v === 'prefers' || v.startsWith('prefers ')) return 'prefer' + trimmed.slice(7);
  if (v === 'likes' || v.startsWith('likes ')) return 'like' + trimmed.slice(5);
  if (v === 'has' || v.startsWith('has ')) return 'have' + trimmed.slice(3);
  if (v === 'is' || v.startsWith('is ')) return 'am' + trimmed.slice(2);
  return verb;
}

/**
 * Expresses project constraints and limitations as affirmative descriptions of actual operational reality,
 * followed by explicit non-support boundaries where supported by evidence.
 * Prevents 0.8B models from misinterpreting or inverting negative constraints.
 */
export function formatAffirmativeConstraint(subject: string, rawConstraint: string): string {
  const lower = rawConstraint.toLowerCase().trim();

  // Pattern A: Match Point / Sports live data constraint -> static simulation reality
  if (
    (lower.includes('live match') || lower.includes('live data')) &&
    (lower.includes('static') || lower.includes('simulat') || lower.includes('never end'))
  ) {
    return `${subject} currently uses static match data and simulated match progression. It does not have live match connectivity.`;
  }

  // Pattern B: Websocket / sync queue constraint
  if (
    (lower.includes('websocket') || lower.includes('real-time') || lower.includes('realtime')) &&
    (lower.includes('offline') || lower.includes('sync queue') || lower.includes('queue'))
  ) {
    return `${subject} currently operates as an offline sync queue. It does not have real-time websocket access or live external sync.`;
  }

  // Pattern C: General "could not get X, so built / uses Y"
  const couldNotMatch = rawConstraint.match(
    /(?:could not|unable to|failed to)\s+(?:get|access|obtain|use)\s+([^,]+),\s*(?:so\s+)?(?:built|created|implemented|used)\s+(.+)/i
  );
  if (couldNotMatch && couldNotMatch[1] && couldNotMatch[2]) {
    const missingFeature = couldNotMatch[1].trim();
    const actualReality = couldNotMatch[2].trim();
    return `${subject} currently uses ${actualReality}. It does not have ${missingFeature} connectivity.`;
  }

  // Pattern D: Explicit negation in raw constraint
  if (lower.startsWith('could not ') || lower.startsWith('unable to ') || lower.startsWith('lacks ')) {
    return `For my project ${subject}, actual reality: operates as a local or static implementation without ${rawConstraint.replace(/^(could not|unable to|lacks)\s+/i, '')}.`;
  }

  return `For my project ${subject}, actual reality: ${rawConstraint}.`;
}

/**
 * Detects whether a topic is an autobiographical/personal life query that asks for personal
 * experiences, private routines, pets, or family rather than general/conceptual or technical themes.
 *
 * Rules:
 * - "My experience with pets" -> autobiographical (true)
 * - "Why do people keep pets?" -> general (false)
 * - "My morning routine" -> autobiographical (true)
 * - "Why are morning routines useful?" -> general (false)
 * - Technical philosophies, coding habits, or remote work reflections are NOT private autobiographical topics.
 */
export function isAutobiographicalTopic(topic: string): boolean {
  if (!topic || typeof topic !== 'string') return false;
  const lower = topic.toLowerCase().trim();

  // 1. Explicitly general / conceptual question patterns are NEVER autobiographical
  if (
    /^(why do people|why are|why is|how do people|what makes|benefits of|advantages of|the science of|the history of|psychology of|tips for|guide to)\b/i.test(
      lower
    ) ||
    /\b(why are morning routines useful|why do people keep pets)\b/i.test(lower)
  ) {
    return false;
  }

  // 2. Technical, software, engineering, or learning methodology topics are NEVER autobiographical
  if (
    /\b(coding|programming|software|developer|engineering|react|typescript|javascript|python|sql|html|css|docker|database|indexeddb|git|api|ai|machine learning|testing|projects|documentation|remote work|time zones)\b/i.test(
      lower
    )
  ) {
    return false;
  }

  // 3. Must have an explicit first-person indicator
  const hasFirstPerson = /\b(my|i|me|myself|mine|how i|where i|why i)\b/i.test(lower) || /^my experience with\b/i.test(lower);
  if (!hasFirstPerson) {
    return false;
  }

  // 4. Must fall within a private, domestic personal life domain:
  // Domain A: Daily routines & private domestic habits
  const isRoutine =
    /\b(morning routine|daily routine|evening routine|sleep routine|coffee habits?|wake up|breakfast|bedtime)\b/i.test(lower) ||
    (/\b(mornings?|evenings?|weekends?|routines?)\b/i.test(lower) && /\b(my|i spend|how i)\b/i.test(lower));

  // Domain B: Pets & domestic animals
  const isPets = /\b(pets?|dogs?|cats?|puppy|puppies|kitten|kittens|owning pets?)\b/i.test(lower);

  // Domain C: Family, romantic relationships & personal childhood history
  const isFamilyOrChildhood =
    /\b(childhood|growing up|my hometown|where i live|family|parents|spouse|partner|dating life|dating|college days|school days)\b/i.test(lower);

  // Domain D: Non-software physical belongings
  const isPersonalBelongings = /\b(my car|my apartment|my house|my wardrobe)\b/i.test(lower);

  // Domain E: Explicit autobiographical experience phrasing
  const isExplicitExperience = /^my experience with (owning|living with|dating|raising)\b/i.test(lower);

  return isRoutine || isPets || isFamilyOrChildhood || isPersonalBelongings || isExplicitExperience;
}

/**
 * Detects whether a statement describes interpersonal communication behavior
 * (debates, discussions, arguments, listening to others, collaboration).
 */
export function isInterpersonalBehavior(predicate: string, value: string): boolean {
  const combined = `${predicate} ${value}`.toLowerCase();
  return /\b(argue|arguing|arguments?|disagree|disagreement|discussions?|debate|debating|debates?|listen|listening|put my point|point of view|viewpoints?|differing opinions)\b/i.test(
    combined
  );
}

/**
 * Formats interpersonal communication habits cleanly and naturally
 * without duplicating predicates or confusing them with future goals or dev workflows.
 */
function formatInterpersonalApproach(predicate: string, value: string): string {
  const cleanPred = predicate.trim();
  const cleanVal = value.trim();
  const lowerPred = cleanPred.toLowerCase();
  const lowerVal = cleanVal.toLowerCase();

  // If value already includes the predicate (e.g. value has "listen to theirs as well")
  if (lowerPred && lowerVal.includes(lowerPred)) {
    const naturalVal = cleanVal.charAt(0).toLowerCase() + cleanVal.slice(1);
    if (lowerVal.startsWith('if ')) {
      return `[Personal Approach]: In discussions, ${naturalVal}.`;
    }
    return `[Personal Approach]: In discussions, I ${naturalVal}.`;
  }

  // If value starts with a condition like "If I have..."
  if (lowerVal.startsWith('if ')) {
    const naturalVal = cleanVal.charAt(0).toLowerCase() + cleanVal.slice(1);
    return `[Personal Approach]: In discussions, ${naturalVal}, I ${toFirstPersonVerb(cleanPred)}.`;
  }

  // If predicate starts with "if "
  if (lowerPred.startsWith('if ')) {
    const naturalPred = cleanPred.charAt(0).toLowerCase() + cleanPred.slice(1);
    return `[Personal Approach]: In discussions, ${naturalPred}, I ${cleanVal}.`;
  }

  const verbPhrase = toFirstPersonVerb(cleanPred);
  if (cleanVal) {
    return `[Personal Approach]: In discussions, I ${verbPhrase} ${cleanVal}.`;
  }
  return `[Personal Approach]: In discussions, I ${verbPhrase}.`;
}

/**
 * Formats a knowledge item into an explicit first-person factual statement.
 * Guarantees that the author ('I') is framed as the direct owner/subject of
 * the project, skill, habit, preference, or goal, preventing 3rd-person hallucination.
 * Strictly omits source excerpts and raw messages.
 */
export function formatKnowledgeItemForGeneration(item: PersonalKnowledgeItem): string {
  const category = (item.category || 'facts').trim().toLowerCase();
  const subject = (item.subject || '').trim();
  const predicate = (item.predicate || '').trim();
  const value = (item.value || '').trim();

  const lowerSubject = subject.toLowerCase();
  const lowerPredicate = predicate.toLowerCase().replace(/_/g, ' ');
  const isAuthor = isAuthorSubject(subject);

  let statement = '';

  // 1. Interpersonal behavior (discussions, debates, arguments) takes precedence
  // regardless of whether legacy extraction tagged it as habits, goals, or opinions.
  if (isInterpersonalBehavior(predicate, value)) {
    statement = formatInterpersonalApproach(predicate, value);
  }
  // 2. If subject already explicitly refers to the author ('I', 'my ...', 'me')
  else if (lowerSubject === 'i' || lowerSubject.startsWith('i ') || lowerSubject.startsWith('my ') || lowerSubject === 'me') {
    statement = `${subject} ${predicate} ${value}`;
  }
  // 3. Projects category
  else if (category === 'projects') {
    if (
      lowerPredicate === 'built' ||
      lowerPredicate === 'created' ||
      lowerPredicate === 'developed' ||
      lowerPredicate === 'made'
    ) {
      let cleanVal = value;
      if (cleanVal) {
        cleanVal = cleanVal
          .replace(/\b(?:(?:that|which)\s+)?(?:i|we)\s+(?:built|created|developed|made)\s+/gi, '')
          .replace(/\s+/g, ' ')
          .trim();
      }
      const projName = isAuthor ? (cleanVal || 'a project') : subject;
      const valueDetails = !isAuthor && cleanVal
        ? cleanVal.toLowerCase().startsWith('a ') ||
          cleanVal.toLowerCase().startsWith('an ') ||
          cleanVal.toLowerCase().startsWith('the ')
          ? `, ${cleanVal}`
          : ` (${cleanVal})`
        : '';
      statement = `[Completed Project]: I built ${projName}${valueDetails}.`;
    } else if (
      lowerPredicate === 'built for' ||
      lowerPredicate === 'created for' ||
      lowerPredicate === 'developed for'
    ) {
      const projName = isAuthor ? (subject && !isAuthorSubject(subject) ? subject : 'my project') : subject;
      statement = `[Project Context]: I built ${projName} for ${value}.`;
    } else if (
      lowerPredicate.includes('difficulty') ||
      lowerPredicate.includes('constraint') ||
      lowerPredicate.includes('limitation') ||
      lowerPredicate.includes('workaround')
    ) {
      const affirmativeReality = formatAffirmativeConstraint(isAuthor ? 'my project' : subject, value);
      statement = `[Project Architecture & Reality]: ${affirmativeReality}`;
    } else {
      statement = isAuthor
        ? `[Project Detail]: I ${toFirstPersonVerb(lowerPredicate)} ${value}.`
        : `[Project Detail]: For my project ${subject}, I ${toFirstPersonVerb(lowerPredicate)} ${value}.`;
    }
  }
  // 4. Skills category
  else if (category === 'skills') {
    if (
      lowerPredicate === 'uses' ||
      lowerPredicate === 'knows' ||
      lowerPredicate === 'learned' ||
      lowerPredicate === 'proficient in' ||
      lowerPredicate === 'skilled in'
    ) {
      const skillName = isAuthor ? value : subject;
      const detail = !isAuthor && value ? ` (${value})` : '';
      statement = `[Current Skill]: I learned and actively use ${skillName}${detail}.`;
    } else {
      statement = isAuthor
        ? `[Current Skill]: I have experience with ${value || predicate}.`
        : `[Current Skill]: I have experience with ${subject}: ${predicate} ${value}.`;
    }
  }
  // 5. Goals category
  else if (category === 'goals') {
    const rawGoal = [lowerPredicate, value].filter(Boolean).join(' ');
    if (isAuthor) {
      const action = rawGoal.replace(/^(plans to|wants to|aims to|planning to|plan to|want to|aim to)\s+/, '');
      statement = `[Future Goal]: My goal is to ${action}.`;
    } else {
      if (rawGoal.startsWith('plans to ') || rawGoal.startsWith('wants to ') || rawGoal.startsWith('aims to ')) {
        const action = rawGoal.replace(/^(plans to|wants to|aims to)\s+/, '');
        statement = `[Future Goal]: My goal is to ${action} (${subject}).`;
      } else {
        statement = `[Future Goal]: My goal for ${subject} is to ${value || predicate}.`;
      }
    }
  }
  // 6. Habits category
  else if (category === 'habits') {
    const cleanPred = predicate.replace(/_/g, ' ');
    const isWorkflow = /\b(code|coding|dev|develop|workflow|git|commit|test|testing|build|deploy)\b/i.test(
      `${subject} ${cleanPred} ${value}`
    );
    if (isWorkflow) {
      statement = isAuthor
        ? `[Work Routine]: In my workflow, I ${toFirstPersonVerb(cleanPred)} ${value}.`
        : `[Work Routine]: In my workflow with ${subject}, I ${toFirstPersonVerb(cleanPred)} ${value}.`;
    } else {
      statement = isAuthor
        ? `[Personal Habit]: I ${toFirstPersonVerb(cleanPred)} ${value}.`
        : `[Personal Habit]: With ${subject}, I ${toFirstPersonVerb(cleanPred)} ${value}.`;
    }
  }
  // 7. Preferences category
  else if (category === 'preferences') {
    statement = isAuthor
      ? `[Personal Preference]: I prefer ${value || predicate}.`
      : `[Personal Preference]: I prefer ${subject}: ${value || predicate}.`;
  }
  // 8. Opinions category
  else if (category === 'opinions') {
    statement = isAuthor
      ? `[Personal Perspective]: My perspective is that ${value || predicate}.`
      : `[Personal Perspective]: On ${subject}, my perspective is that ${value || predicate}.`;
  }
  // 9. General fallback
  else {
    const cleanPred = predicate.replace(/_/g, ' ');
    statement = isAuthor
      ? `[Personal Fact]: I ${toFirstPersonVerb(cleanPred)} ${value}.`
      : `[Personal Fact]: I ${toFirstPersonVerb(cleanPred)} ${subject} ${value}.`;
  }

  // Clean double spaces, punctuation, and enforce character limit
  const cleaned = statement
    .replace(/\s+/g, ' ')
    .replace(/\s+\./g, '.')
    .replace(/\.+$/, '.')
    .trim();

  return cleaned.slice(0, 500);
}

interface ScoredKnowledgeItem {
  item: PersonalKnowledgeItem;
  score: number;
}

/**
 * Calculates a relevance score for a knowledge item against extracted topic keywords.
 * Returns 0 if there are no keyword matches.
 */
export function scoreKnowledgeItem(
  item: PersonalKnowledgeItem,
  keywords: string[]
): number {
  if (keywords.length === 0) {
    return 0;
  }

  const subject = (item.subject || '').toLowerCase().trim();
  const predicate = (item.predicate || '').toLowerCase().trim();
  const value = (item.value || '').toLowerCase().trim();
  const contextScope = (item.contextScope || '').toLowerCase().trim();
  const isAuthor = isAuthorSubject(subject);

  let matchScore = 0;
  let hasMatch = false;
  let subjectMatches = 0;
  let valueMatches = 0;
  let predicateMatches = 0;

  for (const kw of keywords) {
    const kwStem = stemWord(kw);

    // 1. Subject matches (strongest topical anchor when not a generic author placeholder)
    if (!isAuthor && subject) {
      const subMatch = matchKeywordInText(kw, kwStem, subject);
      if (subMatch.exact) {
        matchScore += 8;
        hasMatch = true;
        subjectMatches++;
      } else if (subMatch.stem) {
        matchScore += 6;
        hasMatch = true;
        subjectMatches++;
      }
    }

    // 2. Value matches (details of project, skill, preference)
    if (value) {
      const valMatch = matchKeywordInText(kw, kwStem, value);
      if (valMatch.exact) {
        matchScore += 5;
        hasMatch = true;
        valueMatches++;
      } else if (valMatch.stem) {
        matchScore += 4;
        hasMatch = true;
        valueMatches++;
      }
    }

    // 3. Context scope matches
    if (contextScope) {
      const ctxMatch = matchKeywordInText(kw, kwStem, contextScope);
      if (ctxMatch.exact) {
        matchScore += 4;
        hasMatch = true;
      } else if (ctxMatch.stem) {
        matchScore += 3;
        hasMatch = true;
      }
    }

    // 4. Predicate matches
    // When subject is the author, the predicate is the primary action anchor of the statement
    if (predicate) {
      const predMatch = matchKeywordInText(kw, kwStem, predicate);
      if (predMatch.exact) {
        const points = isAuthor ? 8 : 3;
        matchScore += points;
        hasMatch = true;
        predicateMatches++;
      } else if (predMatch.stem) {
        const points = isAuthor ? 7 : 2;
        matchScore += points;
        hasMatch = true;
        predicateMatches++;
      }
    }
  }

  if (!hasMatch) {
    return 0;
  }

  // Multi-term reinforcement: reward items that match multiple distinct prompt keyword roots
  const matchedRoots = new Set<string>();
  for (const kw of keywords) {
    const kwStem = stemWord(kw);
    const inSub = !isAuthor && subject && (matchKeywordInText(kw, kwStem, subject).exact || matchKeywordInText(kw, kwStem, subject).stem);
    const inVal = value && (matchKeywordInText(kw, kwStem, value).exact || matchKeywordInText(kw, kwStem, value).stem);
    const inPred = predicate && (matchKeywordInText(kw, kwStem, predicate).exact || matchKeywordInText(kw, kwStem, predicate).stem);
    const inCtx = contextScope && (matchKeywordInText(kw, kwStem, contextScope).exact || matchKeywordInText(kw, kwStem, contextScope).stem);

    if (inSub || inVal || (isAuthor && inPred) || inCtx) {
      matchedRoots.add(kwStem);
    }
  }

  if (matchedRoots.size > 1) {
    matchScore += matchedRoots.size * 2;
  }

  // Preference for higher importance (tie-breaker)
  if (item.importance === 'primary') {
    matchScore += 1.0;
  } else if (item.importance === 'secondary') {
    matchScore += 0.5;
  }

  // Preference for higher confidence (tie-breaker)
  if (item.confidence === 'high') {
    matchScore += 0.5;
  } else if (item.confidence === 'medium') {
    matchScore += 0.25;
  }

  return matchScore;
}

/**
 * Checks whether a candidate knowledge item is redundant or a near-duplicate
 * of an already accepted item for the same subject.
 */
function isNearDuplicate(item: PersonalKnowledgeItem, accepted: PersonalKnowledgeItem[]): boolean {
  const normSub = (item.subject || '').toLowerCase().trim();
  const normPred = (item.predicate || '').toLowerCase().trim().replace(/_/g, ' ');
  const normVal = (item.value || '').toLowerCase().trim();

  for (const prev of accepted) {
    const prevSub = (prev.subject || '').toLowerCase().trim();
    const prevPred = (prev.predicate || '').toLowerCase().trim().replace(/_/g, ' ');
    const prevVal = (prev.value || '').toLowerCase().trim();

    if (normSub === prevSub) {
      // 1. Identical value, or one substantive value subsumes the other
      if (normVal && prevVal) {
        if (normVal === prevVal) {
          return true;
        }
        if (prevVal.length >= 4 && normVal.includes(prevVal)) {
          return true;
        }
        if (normVal.length >= 4 && prevVal.includes(normVal)) {
          return true;
        }
      }
      // 2. Same subject and predicate where one or both values are empty
      if (normPred === prevPred && (!normVal || !prevVal)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Retrieves at most 5 active, verified, highly relevant personal knowledge items matching the topic.
 * Applies relative score thresholding and deduplication to avoid context bloat.
 */
export async function retrieveRelevantKnowledge(
  topic: string,
  targetDb: EchoDatabase = db,
  maxItems: number = 5
): Promise<RetrievedKnowledgeResult> {
  const keywords = extractTopicKeywords(topic);

  if (keywords.length === 0) {
    return {
      items: [],
      formattedStrings: [],
      keywordsUsed: [],
      hasRelevantKnowledge: false,
    };
  }

  if (!targetDb.isOpen()) {
    await targetDb.open();
  }

  // 1. Fetch only active items (Dexie indexed on status)
  const activeItems = await getPersonalKnowledge({ status: 'active' }, targetDb);

  // 2. Filter candidates based on confidence and valid supporting evidence
  const candidateItems: PersonalKnowledgeItem[] = [];

  for (const item of activeItems) {
    // Only accept status === 'active'
    if (item.status !== 'active') {
      continue;
    }

    // Exclude low confidence, unrated, or unverified items
    if (item.confidence !== 'high' && item.confidence !== 'medium') {
      continue;
    }

    // Ensure item has an ID to query evidence or participate in retrieval
    if (typeof item.id !== 'number') {
      continue;
    }

    // Verify valid supporting evidence exists for extracted and legacy facts.
    // Facts explicitly marked as manual (sourceType === 'manual') are curated directly by the user
    // and are valid without requiring evidence records.
    // Legacy facts (where sourceType is undefined) are treated as extracted and must pass evidence validation.
    const isManual = item.sourceType === 'manual';

    if (!isManual) {
      const evidenceList: KnowledgeEvidence[] = await getKnowledgeEvidence(item.id, targetDb);
      const hasValidEvidence = evidenceList.some((e) => e.isValid === true);

      if (!hasValidEvidence) {
        // Excluded: archived evidence, invalidated evidence, or orphaned knowledge
        continue;
      }
    }

    candidateItems.push(item);
  }

  // 3. Score candidates against topic keywords
  const scoredItems: ScoredKnowledgeItem[] = [];

  for (const candidate of candidateItems) {
    const score = scoreKnowledgeItem(candidate, keywords);
    if (score > 0) {
      scoredItems.push({ item: candidate, score });
    }
  }

  // 4. Sort by score descending, then by last confirmed date / ID descending
  scoredItems.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    const timeB = b.item.lastConfirmedAt ? new Date(b.item.lastConfirmedAt).getTime() : 0;
    const timeA = a.item.lastConfirmedAt ? new Date(a.item.lastConfirmedAt).getTime() : 0;
    if (timeB !== timeA) {
      return timeB - timeA;
    }
    return (b.item.id ?? 0) - (a.item.id ?? 0);
  });

  // 5. Relative relevance cutoff:
  // When top score is strong (>= 10), reject incidental/weak matches that do not meet the relative threshold.
  const topScore = scoredItems[0]?.score ?? 0;
  const cutoff = topScore >= 10 ? Math.max(7.5, topScore * 0.50) : 3.0;
  const filteredScoredItems = scoredItems.filter((s) => s.score >= cutoff);

  // 6. Deduplicate near-duplicate facts and cap at maxItems (default 5)
  const topItems: PersonalKnowledgeItem[] = [];
  for (const s of filteredScoredItems) {
    if (!isNearDuplicate(s.item, topItems)) {
      topItems.push(s.item);
      if (topItems.length >= maxItems) {
        break;
      }
    }
  }

  const formattedStrings = topItems.map(formatKnowledgeItemForGeneration);

  const hasRelevant = topItems.length > 0;
  const isAutoBio = isAutobiographicalTopic(topic);
  const isAutobiographicalWithoutFacts = !hasRelevant && isAutoBio;
  const safetyNotice = isAutobiographicalWithoutFacts
    ? "No verified personal notes exist for this autobiographical topic. Echo will provide a general reflection to prevent fabricating personal experiences."
    : undefined;

  return {
    items: topItems,
    formattedStrings,
    keywordsUsed: keywords,
    hasRelevantKnowledge: hasRelevant,
    isAutobiographicalWithoutFacts,
    safetyNotice,
  };
}

