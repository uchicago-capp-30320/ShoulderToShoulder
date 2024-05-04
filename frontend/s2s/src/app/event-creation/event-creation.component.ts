import { Component } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';

// services
import { EventService } from '../_services/event.service';

// helpers
import { Event } from '../_models/event';

@Component({
  selector: 'app-event-creation',
  templateUrl: './event-creation.component.html',
  styleUrl: './event-creation.component.css'
})
export class EventCreationComponent {
  showConfirmDialog = false;
  eventForm = new FormGroup({
    title: new FormControl('', Validators.required),
    datetime: new FormControl('', Validators.required),
    duration_h: new FormControl('', Validators.required),
    address: new FormControl('', Validators.required),
    max_attendees: new FormControl('', Validators.required)
  });

  constructor(
    private eventService: EventService
  ) {}

  showConfirmationDialog(): void {
    this.showConfirmDialog = true;
  }

  onSubmit(): void {
    let title = this.eventForm.get('title')?.value;
    let datetime = this.eventForm.get('datetime')?.value;
    let duration_h = this.eventForm.get('duration_h')?.value;
    let address = this.eventForm.get('address')?.value;
    let max_attendees = this.eventForm.get('max_attendees')?.value;

    if (title && datetime && duration_h && address && max_attendees) {
      let newEvent: Event = {
        title: title,
        datetime: datetime,
        duration_h: parseInt(duration_h),
        address: address,
        max_attendees: parseInt(max_attendees)
      };
      this.eventService.createEvent(newEvent).subscribe(() => {
        console.log('Event created successfully');
      });
    }
  }

}
