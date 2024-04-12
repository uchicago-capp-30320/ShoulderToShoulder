// Desc: General constants used throughout the application.

/**
 * A label-value pair with a number value.
 */
export interface labelValueInt {
    label: string;
    value: number;
}

/**
 * A label-value pair with a string value.
 */
export interface labelValueString {
    label: string;
    value: string;
}

/**
 * A label-value pair with an integer array value.
 */
export interface labelValueIntArray {
    label: string;
    value: number[];
}