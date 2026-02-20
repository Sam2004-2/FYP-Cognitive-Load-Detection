import formAData from '../assets/study/paired_associate_form_A.json';
import formBData from '../assets/study/paired_associate_form_B.json';
import { StudyDifficulty, StudyForm, StudyRecognitionChoice, StudyStimulusItem } from '../types/study';

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function stableHash(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function shuffleWithSeed<T>(items: T[], seed: number): T[] {
  const random = seededRandom(seed);
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function getFormData(form: StudyForm): StudyStimulusItem[] {
  const source = form === 'A' ? formAData : formBData;
  return source as StudyStimulusItem[];
}

export function getStimulusItemsForBlock(
  form: StudyForm,
  difficulty: StudyDifficulty,
  count: number,
  participantSeedInput: string
): StudyStimulusItem[] {
  const items = getFormData(form).filter((item) => item.difficulty === difficulty);
  const seed = stableHash(`${participantSeedInput}:${form}:${difficulty}`);
  return shuffleWithSeed(items, seed).slice(0, count);
}

export function buildRecognitionChoices(
  item: StudyStimulusItem,
  allBlockItems: StudyStimulusItem[],
  choiceCount: number,
  useInterferenceDistractors: boolean,
  participantSeedInput: string
): StudyRecognitionChoice[] {
  const poolFromGroup = allBlockItems.filter(
    (candidate) =>
      candidate.id !== item.id && candidate.interferenceGroup === item.interferenceGroup
  );

  const poolAll = allBlockItems.filter((candidate) => candidate.id !== item.id);
  const prioritizedPool = useInterferenceDistractors
    ? [...poolFromGroup, ...poolAll.filter((c) => c.interferenceGroup !== item.interferenceGroup)]
    : poolAll;

  const uniqueDistractors: string[] = [];
  for (const candidate of prioritizedPool) {
    if (candidate.target === item.target) continue;
    if (!uniqueDistractors.includes(candidate.target)) {
      uniqueDistractors.push(candidate.target);
    }
    if (uniqueDistractors.length >= choiceCount - 1) break;
  }

  const randomSeed = stableHash(`${participantSeedInput}:${item.id}:recognition`);
  const shuffledDistractors = shuffleWithSeed(uniqueDistractors, randomSeed);

  const choices: StudyRecognitionChoice[] = [
    { value: item.target, isCorrect: true },
    ...shuffledDistractors.slice(0, Math.max(choiceCount - 1, 1)).map((value) => ({
      value,
      isCorrect: false,
    })),
  ];

  return shuffleWithSeed(choices, randomSeed ^ 0x9e3779b1);
}

export interface DelayedPacket {
  participantId: string;
  sessionNumber: 1 | 2;
  form: StudyForm;
  generatedAtIso: string;
  easyItems: StudyStimulusItem[];
  hardItems: StudyStimulusItem[];
}

export function createDelayedPacket(
  participantId: string,
  sessionNumber: 1 | 2,
  form: StudyForm,
  easyItems: StudyStimulusItem[],
  hardItems: StudyStimulusItem[]
): DelayedPacket {
  return {
    participantId,
    sessionNumber,
    form,
    generatedAtIso: new Date().toISOString(),
    easyItems,
    hardItems,
  };
}
