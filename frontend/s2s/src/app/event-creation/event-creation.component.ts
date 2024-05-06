import { Component, OnInit } from '@angular/core';
import { FormGroup, FormControl, Validators } from '@angular/forms';

// services
import { EventService } from '../_services/event.service';
import { AuthService } from '../_services/auth.service';
import { HobbyService } from '../_services/hobbies.service';

// helpers
import { Event, EventPost } from '../_models/event';
import { states } from '../_helpers/location';
import { labelValueString } from '../_helpers/abstractInterfaces';
import { formControlFieldMap } from '../_helpers/preferences';
import { User } from '../_models/user';
import { HobbyType } from '../_models/hobby';

@Component({
  selector: 'app-event-creation',
  templateUrl: './event-creation.component.html',
  styleUrl: './event-creation.component.css'
})
export class EventCreationComponent implements OnInit {
  showConfirmDialog = false;
  states = states;
  hobbyTypes: HobbyType[] = [];
  user: User = this.authService.userValue;
  event!: EventPost;
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
  sampleEvent: EventPost = {
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
  invalidDialogMessage: string = "Please fill out all required fields.";
  showInvalidDialog = false;
  showError = false;

  constructor(
    private eventService: EventService,
    private authService: AuthService,
    private hobbyService: HobbyService
  ) {
  }

  ngOnInit(): void {
    this.hobbyService.hobbyTypes.subscribe(hobbies => {
      this.hobbyTypes = hobbies;
    });
  }

  resetForm(): void {
    this.eventForm.reset();
  }

  highlightInvalidFields(event: any): void {
    // if the button is disabled, the form is invalid
    this.invalidDialogMessage = '';
    let form: FormGroup = this.eventForm;
    if (event.target.querySelector('button') && event.target.querySelector('button').disabled){
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

    if (title && datetime && duration_h && address1 && max_attendees && city && state && event_type) {
      state = (state as unknown as labelValueString).value;
      datetime = new Date(datetime).toISOString();
      let newEvent: EventPost = {
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

  openConfirmationDialog(): void {
    this.formToEvent();
    this.showConfirmDialog = true;
    console.log(this.eventForm.value);
  }

  onSubmit(): void {
    this.showConfirmDialog = false;
    this.resetForm();
    console.log(this.event)

    // this.eventService.createEvent(this.event).subscribe(() => {
    //   console.log('Event created successfully');
    // });
  }
}