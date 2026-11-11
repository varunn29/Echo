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

export interface RetrievedKnowledgeResult {
  items: PersonalKnowledgeItem[];
  formattedStrings: string[];
  keywordsUsed: string[];
  hasRelevantKnowledge: boolean;
}

/**
 * Extracts meaningful non-stopword tokens from a given topic or text.
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
  const seen = new Set<string>();

  for (const token of rawTokens) {
    if (!COMMON_STOP_WORDS.has(token) && !seen.has(token)) {
      seen.add(token);
      meaningfulKeywords.push(token);
    }
  }

  return meaningfulKeywords;
}

function toFirstPersonVerb(verb: string): string {
  const v = verb.toLowerCase().trim();
  if (v === 'uses') return 'use';
  if (v === 'plans') return 'plan';
  if (v === 'plans to learn') return 'plan to learn';
  if (v === 'wants') return 'want';
  if (v === 'wants to learn') return 'want to learn';
  if (v === 'aims') return 'aim';
  if (v === 'aims to') return 'aim to';
  if (v === 'believes') return 'believe';
  if (v === 'prefers') return 'prefer';
  if (v === 'likes') return 'like';
  if (v === 'has') return 'have';
  if (v === 'is') return 'am';
  return verb;
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

  let statement = '';

  // 1. If subject already explicitly refers to the author ('I', 'my ...', 'me')
  if (lowerSubject === 'i' || lowerSubject.startsWith('i ') || lowerSubject.startsWith('my ')) {
    statement = `${subject} ${predicate} ${value}`;
  }
  // 2. Projects category
  else if (category === 'projects') {
    if (
      lowerPredicate === 'built' ||
      lowerPredicate === 'created' ||
      lowerPredicate === 'developed' ||
      lowerPredicate === 'made'
    ) {
      const valueDetails = value
        ? value.toLowerCase().startsWith('a ') ||
          value.toLowerCase().startsWith('an ') ||
          value.toLowerCase().startsWith('the ')
          ? `, ${value}`
          : ` (${value})`
        : '';
      statement = `[Completed Project]: I built ${subject}${valueDetails}. (Status: Completed by me; not an ongoing attempt or another person's project.)`;
    } else if (
      lowerPredicate.includes('difficulty') ||
      lowerPredicate.includes('constraint') ||
      lowerPredicate.includes('limitation') ||
      lowerPredicate.includes('workaround')
    ) {
      statement = `[Project Constraint & Reality]: For my project ${subject}, ${value}. (Strict reality: This is an offline/static simulation; never claim live, real-time, or external feed connectivity.)`;
    } else {
      statement = `[Project Detail]: For my project ${subject}, I ${toFirstPersonVerb(lowerPredicate)} ${value}.`;
    }
  }
  // 3. Skills category
  else if (category === 'skills') {
    if (
      lowerPredicate === 'uses' ||
      lowerPredicate === 'knows' ||
      lowerPredicate === 'learned' ||
      lowerPredicate === 'proficient in' ||
      lowerPredicate === 'skilled in'
    ) {
      statement = `[Current Mastered Skill]: I already learned and actively use ${subject}${value ? ` (${value})` : ''}. (This is an existing skill, not a future goal.)`;
    } else {
      statement = `[Current Mastered Skill]: I have experience with ${subject}: ${predicate} ${value}. (This is an existing skill, not a future goal.)`;
    }
  }
  // 4. Goals category
  else if (category === 'goals') {
    const goalText = [lowerPredicate, value].filter(Boolean).join(' ');
    if (goalText.startsWith('plans to ') || goalText.startsWith('wants to ') || goalText.startsWith('aims to ')) {
      const action = goalText.replace(/^(plans to|wants to|aims to)\s+/, '');
      statement = `[Future Learning Goal (Not yet started)]: My goal is to ${action} (${subject}). (Strictly a personal learning objective; not a business venture or commercial plan.)`;
    } else {
      statement = `[Future Learning Goal (Not yet started)]: My goal for ${subject} is to ${value || predicate}. (Strictly a personal learning objective; not a business venture or commercial plan.)`;
    }
  }
  // 5. Habits category
  else if (category === 'habits') {
    statement = `[Work Routine]: In my workflow with ${subject}, I ${toFirstPersonVerb(lowerPredicate)} ${value}. (Only use the tools explicitly stated; do not extrapolate unmentioned software or platforms.)`;
  }
  // 6. Preferences category
  else if (category === 'preferences') {
    statement = `[Personal Preference]: I prefer ${subject}: ${value || predicate}.`;
  }
  // 7. Opinions category
  else if (category === 'opinions') {
    statement = `[Personal Perspective]: On ${subject}, my perspective is that ${value || predicate}.`;
  }
  // 8. General fallback
  else {
    statement = `[Personal Fact]: I ${toFirstPersonVerb(lowerPredicate)} ${subject} ${value}.`;
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

  const subject = (item.subject || '').toLowerCase();
  const predicate = (item.predicate || '').toLowerCase();
  const value = (item.value || '').toLowerCase();
  const contextScope = (item.contextScope || '').toLowerCase();

  let matchScore = 0;
  let hasMatch = false;

  for (const kw of keywords) {
    const kwRegex = new RegExp(`\\b${kw}\\b`, 'i');

    // Subject matches (strongest topical grounding)
    if (kwRegex.test(subject)) {
      matchScore += 6;
      hasMatch = true;
    } else if (subject.includes(kw)) {
      matchScore += 2;
      hasMatch = true;
    }

    // Value matches (details of project, skill, preference)
    if (kwRegex.test(value)) {
      matchScore += 5;
      hasMatch = true;
    } else if (value.includes(kw)) {
      matchScore += 2;
      hasMatch = true;
    }

    // Context scope matches
    if (contextScope) {
      if (kwRegex.test(contextScope)) {
        matchScore += 4;
        hasMatch = true;
      } else if (contextScope.includes(kw)) {
        matchScore += 1;
        hasMatch = true;
      }
    }

    // Predicate matches
    if (kwRegex.test(predicate)) {
      matchScore += 3;
      hasMatch = true;
    }
  }

  if (!hasMatch) {
    return 0;
  }

  // Preference for higher importance
  if (item.importance === 'primary') {
    matchScore += 3;
  } else if (item.importance === 'secondary') {
    matchScore += 1;
  }

  // Preference for higher confidence
  if (item.confidence === 'high') {
    matchScore += 2;
  } else if (item.confidence === 'medium') {
    matchScore += 1;
  }

  return matchScore;
}

/**
 * Retrieves at most 10 active, verified, relevant personal knowledge items matching the topic.
 */
export async function retrieveRelevantKnowledge(
  topic: string,
  targetDb: EchoDatabase = db,
  maxItems: number = 10
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

    // Ensure item has an ID to query evidence
    if (typeof item.id !== 'number') {
      continue;
    }

    // Verify valid supporting evidence exists
    const evidenceList: KnowledgeEvidence[] = await getKnowledgeEvidence(item.id, targetDb);
    const hasValidEvidence = evidenceList.some((e) => e.isValid === true);

    if (!hasValidEvidence) {
      // Excluded: archived evidence, invalidated evidence, or orphaned knowledge
      continue;
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

  // 5. Cap at maxItems (up to 10)
  const topItems = scoredItems.slice(0, maxItems).map((s) => s.item);
  const formattedStrings = topItems.map(formatKnowledgeItemForGeneration);

  return {
    items: topItems,
    formattedStrings,
    keywordsUsed: keywords,
    hasRelevantKnowledge: topItems.length > 0,
  };
}

