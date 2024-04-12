// Desc: Utils functions

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