import { Component } from '@angular/core';

// services
import { CalendarService } from '../_services/calendar.service';

// helpers
import { days } from '../_helpers/preferences';

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

  toggleAvailability(slotIndex: number, dayIndex: number): void {
    // Toggle availability for a specific slot and day
    this.calendarService.userAvailability[slotIndex].days[dayIndex] = !this.calendarService.userAvailability[slotIndex].days[dayIndex];
  }
}