import { Component } from '@angular/core';

// services
import { CalendarService } from '../_services/calendar.service';

// helpers
import { days } from '../_helpers/preferences';

/**
 * Component to display and toggle user availability.
 * 
 * This component displays the user's availability schedule and allows toggling 
 * the availability status for specific slots and days.
 * 
 * @example
 * ```
 * <app-availability-display></app-availability-display>
 * ```
 * 
 * @remarks
 * This component relies on the CalendarService to manage user availability data.
 * 
 * @see CalendarService
 * 
 * @export
 * @class AvailabilityDisplayComponent
 */
@Component({
  selector: 'app-availability-display',
  templateUrl: './availability-display.component.html',
  styleUrl: './availability-display.component.css'
})
export class AvailabilityDisplayComponent {
  days = days;

  constructor(
    public calendarService: CalendarService
  ) { }

  /**
   * Toggles the availability status for a specific slot and day.
   * 
   * @param {number} slotIndex The index of the availability slot.
   * @param {number} dayIndex The index of the day.
   * @memberof AvailabilityDisplayComponent
   */
  toggleAvailability(slotIndex: number, dayIndex: number): void {
    this.calendarService.userAvailability[slotIndex].days[dayIndex] = 
      !this.calendarService.userAvailability[slotIndex].days[dayIndex];
  }
}