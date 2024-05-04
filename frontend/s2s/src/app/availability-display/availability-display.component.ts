import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { Router } from '@angular/router';

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
export class AvailabilityDisplayComponent implements OnChanges {
  @Input() isEditable: boolean = true;
  @Input() profileView: boolean = false;
  days = days;
  availabilityBackup: any[] = [];

  constructor(
    public calendarService: CalendarService,
    public onboardingService: OnboardingService,
    private router: Router
  ) {
   }
  
  ngOnChanges(changes: SimpleChanges): void {
    this.copyAvailability();
  }

  /**
   * Copies the user's availability data to a backup array.
   */
  copyAvailability(): void {
    this.calendarService.userAvailability$.subscribe(availability => {
      this.availabilityBackup = availability;
      console.log(this.availabilityBackup)
    });
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

  /**
   * Submits the availability form to the server.
   */
  submitAvailability(): void {
    this.isEditable = false;
    this.onboardingService.submitAvailabilityForm().subscribe(() => {
      this.copyAvailability();
    });
  }

  /**
   * Cancels the availability form and resets the availability data.
   */
  cancelAvailability(): void {
    this.isEditable = false;
    this.calendarService.setAvailability(this.availabilityBackup);
    this.router.navigate(['/profile', 2]);
    location.reload();
  }
}