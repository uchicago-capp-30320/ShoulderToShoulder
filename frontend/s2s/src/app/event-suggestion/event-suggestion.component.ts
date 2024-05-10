import { Component, OnInit } from '@angular/core';

// services
import { SuggestionService } from '../_services/suggestion.service';
import { AuthService } from '../_services/auth.service';

// helpers
import { Suggestion } from '../_models/suggestions';

@Component({
  selector: 'app-event-suggestion',
  templateUrl: './event-suggestion.component.html',
  styleUrl: './event-suggestion.component.css'
})
export class EventSuggestionComponent implements OnInit {
  showEventSuggestionDialog = false;
  currentSuggestion!: Suggestion;
  eventSuggestions: Suggestion[] = [];
  
  constructor (
    private suggestionService: SuggestionService,
    private authService: AuthService,
  ) { }

  ngOnInit(): void {
    this.suggestionService.getSuggestions().subscribe(suggestions => {
      if (suggestions.length > 0) {
        this.eventSuggestions = suggestions;
        this.currentSuggestion = suggestions[0];
        this.showEventSuggestionDialog = true;
      }
    });
  }

  onSubmitRSVP(rsvp: string) {
    this.suggestionService.sendRSVP({
      user_id: this.authService.userValue.id,
      event_id: this.currentSuggestion.event_id,
      rsvp: rsvp,
    }).subscribe(() => {
      this.eventSuggestions.shift();
      if (this.eventSuggestions.length > 0) {
        this.currentSuggestion = this.eventSuggestions[0];
      } else {
        this.showEventSuggestionDialog = false;
      }
    });
  }

}
