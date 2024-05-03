import { Component, Input, OnInit } from '@angular/core';

// services
import { CalendarService } from '../_services/calendar.service';
import { OnboardingService } from '../_services/onboarding.service';

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
export class AvailabilityDisplayComponent implements OnInit {
  @Input() isEditable: boolean = true;
  @Input() profileView: boolean = false;
  days = days;
  availabilityBackup: any[] = [];

  constructor(
    public calendarService: CalendarService,
    public onboardingService: OnboardingService
  ) {
   }
  
  ngOnInit(): void {
    this.copyAvailability();
  }

  /**
   * Copies the user's availability data to a backup array.
   */
  copyAvailability(): void {
    for (let i = 0; i < this.calendarService.userAvailability.length; i++) {
      this.availabilityBackup.push({ ...this.calendarService.userAvailability[i] });
    }
    console.log(this.availabilityBackup)
  }

  /**
   * Toggles the availability status for a specific slot and day.
   * 
   * @param {number} slotIndex The index of the availability slot.
   * @param {number} dayIndex The index of the day.
   * @memberof AvailabilityDisplayComponent
   */
  toggleAvailability(slotIndex: number, dayIndex: number): void {
    if (this.isEditable) {
      this.calendarService.userAvailability[slotIndex].days[dayIndex] = 
      !this.calendarService.userAvailability[slotIndex].days[dayIndex];
    }
  }

  submitAvailability(): void {
    this.onboardingService.submitAvailabilityForm();
    this.isEditable = false;
    this.availabilityBackup = this.calendarService.userAvailability;
  }

  cancelAvailability(): void {
    this.copyAvailability();
    this.isEditable = false;
  }
}