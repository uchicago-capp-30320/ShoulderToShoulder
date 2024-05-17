import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';


// services
import { AvailabilityService } from '../_services/availability.service';
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
 * This component relies on the AvailabilityService to manage user availability data.
 *
 * @see AvailabilityService
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
  showLoadingDialog: boolean = false;
  selectedSlots: Set<string> = new Set();
  lastSelectedIndex: string | null = null;


  constructor(
    public availabilityService: AvailabilityService,
    private router: Router,
    public messageService: MessageService
  ) {
   }

  ngOnChanges(changes: SimpleChanges): void {
    this.copyAvailability();
  }

  /**
   * Copies the user's availability data to a backup array.
   */
  copyAvailability(): void {
    this.availabilityService.userAvailability$.subscribe(availability => {
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
      this.availabilityService.userAvailability[slotIndex].days[dayIndex] =
      !this.availabilityService.userAvailability[slotIndex].days[dayIndex];
    }
  }

  getSlotKey(slotIndex: number, dayIndex: number): string {
    return `${slotIndex}-${dayIndex}`;
  }

  /**
   * Clears all messages from the message service.
   */
  clearMessages() {
    this.messageService.clear();
  }

  /**
   * Submits the availability form to the server.
   */
  submitAvailability(): void {
    this.showLoadingDialog = true;
    this.availabilityService.submitAvailability().subscribe(
      data => {
        this.showLoadingDialog = false;
        this.isEditable = false;
        this.clearMessages();
        this.messageService.add({severity: 'success', detail: 'Availability updated successfully!'});
      },
      error => {
        this.showLoadingDialog = false;
        this.isEditable = false;
        console.log(error);
        this.clearMessages();
        this.messageService.add({severity: 'error',
          detail: 'There was an error updating the event. Please try again.'});
        this.showLoadingDialog = false;
      })
  }

  /**
   * Cancels the availability form and resets the availability data.
   */
  cancelAvailability(): void {
    this.isEditable = false;
    this.availabilityService.setAvailability(this.availabilityBackup);
    location.reload();
    this.router.navigate(['/profile/2']);
  }
}
