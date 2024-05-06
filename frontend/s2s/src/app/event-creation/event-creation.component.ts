import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';

// services
import { EventService } from '../_services/event.service';
import { AuthService } from '../_services/auth.service';
import { HobbyService } from '../_services/hobbies.service';
import { MessageService } from 'primeng/api';

// helpers
import { Event } from '../_models/event';
import { states } from '../_helpers/location';
import { formControlFieldMap } from '../_helpers/preferences';
import { User } from '../_models/user';
import { HobbyType } from '../_models/hobby';

/**
 * Component for creating an event.
 * 
 * This component allows users to create an event by filling out a form with
 * event details. The form includes fields for the event title, description,
 * hobby type, date and time, duration, address, city, state, and maximum number
 * of attendees. Users can also choose to add themselves to the event as an
 * attendee.
 * 
 * @example
 * <app-event-creation></app-event-creation>
 * 
 * @see EventService
 * @see AuthService
 * @see HobbyService
 */
@Component({
  selector: 'app-event-creation',
  templateUrl: './event-creation.component.html',
  styleUrl: './event-creation.component.css'
})
export class EventCreationComponent implements OnInit {
  user: User = this.authService.userValue;

  // dialog flags
  showConfirmDialog = false;
  showLoadingDialog = false;

  // event data
  states = states;
  hobbyTypes: HobbyType[] = [];
  event!: Event;
  eventForm = new FormGroup({
    title: new FormControl('', Validators.required),
    description: new FormControl(''),
    hobby_type: new FormControl('', Validators.required),
    datetime: new FormControl('', Validators.required),
    duration_h: new FormControl('', [
      Validators.required,
      Validators.min(1),
      Validators.max(8)]),
    address1: new FormControl('', Validators.required),
    address2: new FormControl(''),
    city: new FormControl('', Validators.required),
    state: new FormControl('', Validators.required),
    max_attendees: new FormControl('', [
      Validators.required,
      Validators.min(1),
      Validators.max(20)]),
    add_user: new FormControl(true)
  });

  // sample event
  sampleEvent: Event = {
    title: 'Cozy Knitting Circle',
    hobby_type: "CRAFTING",
    description: "Join us for a cozy knitting circle at the Bourgeois Pig Cafe! Whether you're a beginner or an experienced knitter, bring your yarn and needles and enjoy a relaxing afternoon of knitting, coffee, and conversation. All skill levels are welcome!",
    datetime: 'May 14, 2024 at 05:30 PM',
    duration_h: 2,
    address1: '738 W Fullerton Ave',
    city: 'Chicago',
    state: 'Illinois',
    max_attendees: 5,
  };
  dateFormat = 'yyyy-MM-ddTHH:mm:ss';

  // error handling
  errorMessage = "Please fill out all required fields."
  errorHeader = "Required Fields Missing"
  invalidDialogMessage: string = "Please fill out all required fields.";
  showInvalidDialog = false;
  showError = false;

  constructor(
    private eventService: EventService,
    private authService: AuthService,
    private hobbyService: HobbyService,
    public messageService: MessageService
  ) {
  }

  ngOnInit(): void {
    this.hobbyService.hobbyTypes.subscribe(hobbies => {
      this.hobbyTypes = hobbies;
    });
  }

  /**
   * Resets the form to its initial state.
   */
  resetForm(): void {
    this.eventForm.reset();
  }

  /**
   * Clears all messages from the message service.
   */
  clearMessages() {
    this.messageService.clear();
  }

  /**
   * Displays an error message if the form is invalid.
   * 
   * @param event The event that triggered the form submission.
   */
  highlightInvalidFields(event: any): void {
    // if the button is disabled, the form is invalid
    let form: FormGroup = this.eventForm;
    if (event.target.querySelector('button') && event.target.querySelector('button').disabled){
      this.messageService.add({severity: 'error', detail: 'Please fill out all required fields.'})
      this.invalidDialogMessage = '';
      this.errorHeader = "Required Fields Missing"
      
      // mark all invalid fields as dirty
      for (let control in form.controls) {
        let formControl = form.controls[control];
        if (formControl.invalid) {
          formControl.markAsDirty();
        }
      }
      this.showInvalidDialog = form.invalid ? true : false;
      this.invalidDialogMessage += " The following fields are missing values: "

      // get the missing fields
      let missingFields = [];
      for (let control in form.controls) {
        let formControl = form.controls[control]
        if (formControl.invalid) {
          missingFields.push(formControlFieldMap[control]);
        }
      }

      this.invalidDialogMessage += missingFields.join(", ");
    }
  }

  /**
   * Converts the form data to an event object.
   */
  formToEvent(): void {
    let title = this.eventForm.get('title')?.value;
    let description = this.eventForm.get('description')?.value;
    let event_type = this.eventForm.get('hobby_type')?.value;
    let datetime = this.eventForm.get('datetime')?.value;
    let duration_h = this.eventForm.get('duration_h')?.value;
    let address1 = this.eventForm.get('address1')?.value;
    let address2 = this.eventForm.get('address2')?.value;
    let city = this.eventForm.get('city')?.value;
    let state = this.eventForm.get('state')?.value;
    let max_attendees = this.eventForm.get('max_attendees')?.value;
    let add_user = this.eventForm.get('add_user')?.value;

    // check if all required fields are filled out
    if (title && datetime && duration_h && address1 && max_attendees && city && state && event_type) {
      datetime = new Date(datetime).toISOString();
      let newEvent: Event = {
        title: title,
        created_by: this.user.id,
        description: description ? description : '',
        hobby_type: event_type ? (event_type as unknown as HobbyType).type: "OTHER",
        datetime: datetime,
        duration_h: parseInt(duration_h),
        address1: address1,
        address2: address2 ? address2 : '',
        city: city,
        state: state,
        max_attendees: parseInt(max_attendees),
        add_user: add_user ? add_user : false
      };
      this.event = newEvent;
    }
  }

  /**
   * Opens the confirmation dialog.
   */
  openConfirmationDialog(): void {
    this.formToEvent();
    this.showConfirmDialog = true;
    console.log(this.event);
  }

  /**
   * Submits the event creation form.
   */
  onSubmit(): void {
    this.showConfirmDialog = false;
    this.showLoadingDialog = true;
    console.log(this.event)

    // create the event using the event service
    this.eventService.createEvent(this.event).subscribe(
      data => {
        // if there were no errors, display a success message and reset form
        console.log(data);
        this.clearMessages();
        this.messageService.add({severity: 'success', detail: 'Event created successfully!'});
        this.showLoadingDialog = false;
        this.resetForm();
      },
      error => {
        // if there was an error, display an error message
        console.log(error);
        this.clearMessages();
        this.messageService.add({severity: 'error', 
          detail: 'There was an error creating the event. Please try again.'});
        this.showLoadingDialog = false;
        this.invalidDialogMessage = "There was an error creating an event. The following error occurred: " + error.error.error;
        this.errorHeader = "Error Creating Event";
        this.showInvalidDialog = true;
      }
    );
  }
}
