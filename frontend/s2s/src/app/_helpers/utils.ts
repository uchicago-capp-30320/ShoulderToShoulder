// Purpose: Common utility functions.
import { states } from "./location";
import { labelValueString } from "./abstractInterfaces";

/**
 * Get random integer between min and max.
 *
 * @param min The minimum value.
 * @param max The maximum value.
 * @returns The random integer.
 */
export function getRandomInt(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Get a random subset of an array.
 *
 * @param arr The array to get a subset of.
 * @param size The size of the subset.
 */
export function getRandomSubset(arr: any[], size: number) {
  // Shuffle the array and return the first size elements.
  let shuffled = shuffleArray(arr);

  // Return the first size elements.
  return shuffled.slice(0, size);
}

/**
 * Shuffle an array.
 *
 * @param arr The array to shuffle.
 * @returns The shuffled array.
 */
export function shuffleArray(arr: any[]) {
  let shuffled = arr.slice(0), i = arr.length, temp, index;

  // while there remain elements to shuffle...
  while (i--) {
    // pick a remaining element and swap it with the current element.
    index = Math.floor((i + 1) * Math.random());
    temp = shuffled[index];
    shuffled[index] = shuffled[i];
    shuffled[i] = temp;
  }

  return shuffled;
}

/**
 * Split a string at the first digit and capitalize the first letter of the prefix.
 *
 * @param input The input string.
 * @returns The formatted string.
 */
export function splitString(input: string): string {
  const index = input.search(/\d/);  // find the index of the first digit
  if (index === -1) {
      return input;  // return the original input if no digits are found
  }

  // split the string at the found index
  let prefix = input.substring(0, index);
  let suffix = input.substring(index);

  // capitalize the first letter of the prefix if necessary
  prefix = prefix.charAt(0).toUpperCase() + prefix.slice(1);

  return prefix + " " + suffix;
}

/**
 * Create an array range of numbers.
 *
 * @param start The start of the range (inclusive).
 * @param end The end of the range (exclusive).
 * @returns The array range.
 */
export function range(start: number, end: number): number[] {
  return Array.from({ length: end - start }, (_, i) => start + i);
}

/**
 * Gets a labelValueString object for a state based on the state's name or
 * abbreviation.
 */
export function getState(state: string): labelValueString {
  let stateObj = states.find(s => s.label === state || s.value === state);
  return stateObj ? stateObj : { label: '', value: '' };
}
